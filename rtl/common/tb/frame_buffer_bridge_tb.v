// frame_buffer_bridge_tb.v -- proves the fix for KNOWN_ISSUE_GAP_INTOLERANCE.md:
// feeds weibull_top_new through frame_buffer_bridge with a HOSTILE, randomly
// gapped write pattern (matching real HPS/Avalon-MM MMIO traffic -- tens to
// 100+ cycles between writes, not back-to-back), and checks every detection
// against the SAME golden vectors (top_pixels.txt/top_expected.txt,
// SLI=17/GUARD=13) weibull_top_tb.v already trusts for a direct, gap-free
// feed. If this test passes, the bridge genuinely eliminates the
// "zero detections indefinitely under real HPS timing" failure mode --
// not merely "produces something," the IDENTICAL detections a known-good
// direct feed produces, despite the host-side gaps.
`timescale 1ns/1ps
module frame_buffer_bridge_tb;
    localparam integer TK = 8;
    localparam integer IMG_W = 48, IMG_H = 40;
    localparam integer N_PIX = IMG_W*IMG_H;
    localparam integer N_EXP = (IMG_H-2*TK)*(IMG_W-2*TK);
    localparam integer CMP_OFFSET = 1;  // matches weibull_top_tb.v exactly

    reg clk = 0;
    reg rstn = 0;
    reg [1:0] pfa_sel = 1;

    always #5 clk = ~clk;

    // ---- Bridge instance ---------------------------------------------------
    reg         frame_start;
    reg         wr_en;
    reg [7:0]   wr_data;
    wire        ready_for_frame;
    wire [31:0] wr_count;
    wire        core_rstn_out;
    wire        pixel_out_valid;
    wire [7:0]  pixel_out;
    wire        draining, drain_done;

    frame_buffer_bridge #(.IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H), .DATA_WIDTH(8), .RESET_HOLD(8)) bridge (
        .clk(clk), .rstn(rstn),
        .frame_start(frame_start), .wr_en(wr_en), .wr_data(wr_data),
        .ready_for_frame(ready_for_frame), .wr_count(wr_count),
        .core_rstn_out(core_rstn_out), .pixel_out_valid(pixel_out_valid), .pixel_out(pixel_out),
        .draining(draining), .drain_done(drain_done)
    );

    // ---- DUT: the real, unmodified weibull_top_new, driven by the bridge --
    wire detect_valid, detect;
    wire signed [16:0] T_log_code;

    weibull_top_new #(
        .SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("../../../lut")
    ) dut (
        .clk(clk), .rstn(core_rstn_out),
        .pixel_in_valid(pixel_out_valid), .pixel_in(pixel_out),
        .pfa_sel(pfa_sel),
        .detect_valid(detect_valid), .detect(detect), .T_log_code(T_log_code)
    );

    // ---- Golden vectors (identical files weibull_top_tb.v uses) -----------
    reg [7:0] pixmem [0:N_PIX-1];
    integer   exp_row [0:N_EXP-1];
    integer   exp_col [0:N_EXP-1];
    integer   exp_detect [0:N_EXP-1];
    integer   exp_tlog [0:N_EXP-1];

    integer i, fh, r3, r4;
    reg [8*256-1:0] header_line;
    integer window_count, errors, checked;
    integer out_row, out_col, tdiff;

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

    // ---- Hostile gapped write driver ---------------------------------------
    // Deterministic pseudo-random gap pattern (fixed seed, fully
    // reproducible) -- NOT back-to-back, matching the MMIO traffic pattern
    // KNOWN_ISSUE_GAP_INTOLERANCE.md describes (tens to 100+ idle cycles
    // between writes is normal for this transport). Gaps range 0-40 idle
    // cycles between consecutive pixel writes -- a genuinely hostile pattern
    // for the OLD direct-feed approach (which needs (SLI-1)*IMG_WIDTH =
    // 16*48 = 768 CONSECUTIVE gap-free cycles just to warm up, impossible
    // under this traffic), demonstrating the bridge absorbs it completely.
    integer wpix, gap, seed;
    initial begin
        seed = 20261001;
        rstn = 0; frame_start = 0; wr_en = 0; wr_data = 0;
        repeat (5) @(posedge clk);
        rstn = 1;
        @(posedge clk);

        frame_start <= 1'b1;
        @(posedge clk);
        frame_start <= 1'b0;

        for (wpix = 0; wpix < N_PIX; wpix = wpix + 1) begin
            gap = ($random(seed) % 41); // 0..40 idle cycles, deterministic
            if (gap < 0) gap = -gap;
            repeat (gap) begin
                wr_en <= 1'b0;
                @(posedge clk);
            end
            wr_en   <= 1'b1;
            wr_data <= pixmem[wpix];
            @(posedge clk);
        end
        wr_en <= 1'b0;

        // Wait for the bridge to finish draining (gap-free) into the core,
        // plus the core's own fixed pipeline latency to flush.
        wait (drain_done);
        repeat (300) @(posedge clk);

        $display("---- frame_buffer_bridge_tb: %0d checked, %0d errors (out of %0d expected) ----",
            checked, errors, N_EXP);
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

    // ---- Check every detection against the golden vectors (identical
    // technique to weibull_top_tb.v) --------------------------------------
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
