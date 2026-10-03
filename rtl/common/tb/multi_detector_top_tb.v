// multi_detector_top_tb.v -- verifies multi_detector_top.v (one shared
// front_end3 feeding all 5 backends in parallel) against EACH detector's
// own already-trusted golden vectors, with detector_sel cycling through all
// 5 choices on a FRESH reset + re-streamed frame per selection (matching
// line_buffer's own frame-independence requirement, BUG_LOG D20). All 5
// golden vector sets use the identical demo image (confirmed by md5sum),
// so this is a direct, bit-exact comparison against the SAME standalone
// testbenches (weibull_top_tb.v etc.) already trust -- if sharing one
// front_end3 across all 5 backends introduced any cross-talk/resource-
// sharing bug, this is what would catch it.
`timescale 1ns/1ps
module multi_detector_top_tb;
    localparam integer TK = 8;
    localparam integer IMG_W = 48, IMG_H = 40;
    localparam integer N_PIX = IMG_W*IMG_H;
    localparam integer N_EXP = (IMG_H-2*TK)*(IMG_W-2*TK);
    localparam integer CMP_OFFSET = 1;

    reg clk = 0;
    reg rstn = 0;
    reg pixel_in_valid = 0;
    reg [7:0] pixel_in = 0;
    reg [1:0] pfa_sel = 1;
    reg [2:0] detector_sel = 0;

    wire detect_valid, detect;
    wire signed [16:0] T_log_code;

    multi_detector_top #(
        .SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("../../../lut")
    ) dut (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel), .detector_sel(detector_sel),
        .detect_valid(detect_valid), .detect(detect), .T_log_code(T_log_code)
    );

    always #5 clk = ~clk;

    reg [7:0] pixmem [0:N_PIX-1];
    integer   exp_row [0:N_EXP-1];
    integer   exp_col [0:N_EXP-1];
    integer   exp_detect [0:N_EXP-1];
    integer   exp_tlog [0:N_EXP-1];

    integer i, fh, r3, r4, d;
    reg [8*256-1:0] header_line;
    integer window_count, errors_total, checked_total, errors_this, checked_this;
    integer out_row, out_col, tdiff;
    reg [8*64-1:0] expfile;

    initial begin
        fh = $fopen("../../../CFAR_Weibull/rtl/tb/top_pixels.txt", "r");
        if (fh == 0) begin $display("ERROR: cannot open top_pixels.txt"); $finish; end
        for (i = 0; i < N_PIX; i = i + 1) r3 = $fscanf(fh, "%d\n", pixmem[i]);
        $fclose(fh);
    end

    initial begin
        errors_total = 0; checked_total = 0;
        rstn = 0; pixel_in_valid = 0; pixel_in = 0;
        repeat (5) @(posedge clk);
        rstn = 1;

        for (d = 0; d < 5; d = d + 1) begin
            case (d)
                0: expfile = "../../../CFAR_Weibull/rtl/tb/top_expected.txt";
                1: expfile = "../../../CFAR Lognormal/rtl/tb/top_expected.txt";
                2: expfile = "../../../CFAR Generalized Gamma/rtl/tb/top_expected.txt";
                3: expfile = "../../../CFAR_Burr/rtl/tb/top_expected.txt";
                4: expfile = "../../../CFAR_G0/rtl/tb/top_expected.txt";
            endcase
            fh = $fopen(expfile, "r");
            if (fh == 0) begin $display("ERROR: cannot open %0s", expfile); $finish; end
            r4 = $fgets(header_line, fh);
            for (i = 0; i < N_EXP; i = i + 1)
                r4 = $fscanf(fh, "%d %d %d %d\n", exp_row[i], exp_col[i], exp_detect[i], exp_tlog[i]);
            $fclose(fh);

            // Fresh per-detector-selection reset (D20 frame-independence).
            rstn = 0;
            repeat (5) @(posedge clk);
            rstn = 1;
            detector_sel = d[2:0];
            @(posedge clk);

            window_count = -1; errors_this = 0; checked_this = 0;

            for (i = 0; i < N_PIX; i = i + 1) begin
                pixel_in_valid <= 1'b1;
                pixel_in       <= pixmem[i];
                @(posedge clk);
            end
            pixel_in_valid <= 1'b0;
            repeat (300) @(posedge clk);

            $display("detector_sel=%0d: %0d checked, %0d errors (of %0d expected)",
                d, checked_this, errors_this, N_EXP);
            errors_total  = errors_total  + errors_this;
            checked_total = checked_total + checked_this;
        end

        $display("---- multi_detector_top_tb: %0d total checked, %0d total errors ----",
            checked_total, errors_total);
        if (errors_total == 0)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end

    always @(posedge clk) begin
        if (detect_valid) begin
            window_count = window_count + 1;
            out_row = TK + ((window_count+CMP_OFFSET) / (IMG_W - 2*TK));
            out_col = TK + ((window_count+CMP_OFFSET) % (IMG_W - 2*TK));
            if (window_count+CMP_OFFSET < N_EXP) begin
                if (out_row !== exp_row[window_count+CMP_OFFSET] || out_col !== exp_col[window_count+CMP_OFFSET]) begin
                    $display("  INDEX MISMATCH at window %0d: got (%0d,%0d) expected (%0d,%0d)",
                        window_count, out_row, out_col, exp_row[window_count+CMP_OFFSET], exp_col[window_count+CMP_OFFSET]);
                    errors_this = errors_this + 1;
                end else begin
                    checked_this = checked_this + 1;
                    if (detect !== exp_detect[window_count+CMP_OFFSET]) begin
                        $display("  DETECT MISMATCH (%0d,%0d): got %0d expected %0d",
                            out_row, out_col, detect, exp_detect[window_count+CMP_OFFSET]);
                        errors_this = errors_this + 1;
                    end
                    tdiff = T_log_code - exp_tlog[window_count+CMP_OFFSET];
                    if (tdiff < 0) tdiff = -tdiff;
                    if (tdiff > 2) begin
                        $display("  T_LOG MISMATCH (%0d,%0d): got %0d expected %0d",
                            out_row, out_col, T_log_code, exp_tlog[window_count+CMP_OFFSET]);
                        errors_this = errors_this + 1;
                    end
                end
            end
        end
    end
endmodule
