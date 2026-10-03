// weibull_top_tb.v -- full-chain integration test, same pattern as
// gengamma_top_tb.v (see that file's header for the rationale).
`timescale 1ns/1ps
module weibull_top_tb;
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

    wire detect_valid, detect;
    wire signed [16:0] T_log_code;

    weibull_top_new #(
        .SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("../../../lut")
    ) dut (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel),
        .detect_valid(detect_valid), .detect(detect), .T_log_code(T_log_code)
    );

    always #5 clk = ~clk;

    reg [7:0] pixmem [0:N_PIX-1];
    integer   exp_row [0:N_EXP-1];
    integer   exp_col [0:N_EXP-1];
    integer   exp_detect [0:N_EXP-1];
    integer   exp_tlog [0:N_EXP-1];

    integer i, fh, r3, r4;
    reg [8*256-1:0] header_line;
    integer col_cnt, row_cnt;
    integer errors, checked;
    integer window_count;
    integer out_row, out_col;
    integer tdiff;

    initial begin
        fh = $fopen("top_pixels.txt", "r");
        if (fh == 0) begin $display("ERROR: cannot open top_pixels.txt"); $finish; end
        for (i = 0; i < N_PIX; i = i + 1) r3 = $fscanf(fh, "%d\n", pixmem[i]);
        $fclose(fh);

        fh = $fopen("top_expected.txt", "r");
        if (fh == 0) begin $display("ERROR: cannot open top_expected.txt"); $finish; end
        r4 = $fgets(header_line, fh);
        for (i = 0; i < N_EXP; i = i + 1)
            r4 = $fscanf(fh, "%d %d %d %d\n", exp_row[i], exp_col[i], exp_detect[i], exp_tlog[i]);
        $fclose(fh);
        $display("Loaded %0d pixels, %0d expected windows.", N_PIX, N_EXP);
    end

    initial begin
        rstn = 0; pixel_in_valid = 0; pixel_in = 0;
        repeat (5) @(posedge clk);
        rstn = 1;
        @(posedge clk);
        for (row_cnt = 0; row_cnt < IMG_H; row_cnt = row_cnt + 1) begin
            for (col_cnt = 0; col_cnt < IMG_W; col_cnt = col_cnt + 1) begin
                pixel_in_valid <= 1'b1;
                pixel_in       <= pixmem[row_cnt*IMG_W + col_cnt];
                @(posedge clk);
            end
        end
        pixel_in_valid <= 1'b0;
        repeat (200) @(posedge clk);
        $display("---- weibull_top_tb: %0d checked, %0d errors ----", checked, errors);
        if (errors == 0 && checked == N_EXP-CMP_OFFSET)
            $display("PASS");
        else
            $display("FAIL (checked %0d of %0d expected)", checked, N_EXP);
        $finish;
    end

    initial begin
        window_count = -1;
        errors = 0; checked = 0;
    end

    always @(posedge clk) begin
        if (detect_valid) begin
            window_count = window_count + 1;
            out_row = TK + ((window_count+CMP_OFFSET) / (IMG_W - 2*TK));
            out_col = TK + ((window_count+CMP_OFFSET) % (IMG_W - 2*TK));
            if (window_count+CMP_OFFSET < N_EXP) begin
                if (out_row !== exp_row[window_count+CMP_OFFSET] || out_col !== exp_col[window_count+CMP_OFFSET]) begin
                    $display("INDEX MISMATCH at window %0d: got (%0d,%0d) expected (%0d,%0d)",
                        window_count, out_row, out_col, exp_row[window_count+CMP_OFFSET], exp_col[window_count+CMP_OFFSET]);
                    errors = errors + 1;
                end else begin
                    checked = checked + 1;
                    if (detect !== exp_detect[window_count+CMP_OFFSET]) begin
                        $display("DETECT MISMATCH (%0d,%0d): got %0d expected %0d",
                            out_row, out_col, detect, exp_detect[window_count+CMP_OFFSET]);
                        errors = errors + 1;
                    end
                    tdiff = T_log_code - exp_tlog[window_count+CMP_OFFSET];
                    if (tdiff < 0) tdiff = -tdiff;
                    if (tdiff > 2) begin
                        $display("T_LOG MISMATCH (%0d,%0d): got %0d expected %0d",
                            out_row, out_col, T_log_code, exp_tlog[window_count+CMP_OFFSET]);
                        errors = errors + 1;
                    end
                end
            end
        end
    end
endmodule
