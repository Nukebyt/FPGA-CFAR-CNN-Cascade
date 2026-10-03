// gengamma_isolate_tb.v -- diagnostic only: standalone gengamma_top vs.
// multi_detector_top's internal gengamma branch (hierarchical reference),
// fed the identical stream, diffed cycle-by-cycle. Isolates whether
// gengamma_backend misbehaves specifically when co-instantiated with
// burr_backend/g0_backend (which also use log2_lzc), or whether the bug is
// in multi_detector_top's own wiring/mux.
`timescale 1ns/1ps
module gengamma_isolate_tb;
    localparam integer IMG_W = 48, IMG_H = 40;
    localparam integer N_PIX = IMG_W*IMG_H;

    reg clk = 0;
    reg rstn = 0;
    reg pixel_in_valid = 0;
    reg [7:0] pixel_in = 0;
    reg [1:0] pfa_sel = 1;

    always #5 clk = ~clk;

    wire gg_dv, gg_d; wire signed [16:0] gg_tlog;
    gengamma_top #(.SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("../../../lut")) standalone (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel),
        .detect_valid(gg_dv), .detect(gg_d), .T_log_code(gg_tlog)
    );

    wire md_dv, md_d; wire signed [16:0] md_tlog;
    reg [2:0] detector_sel = 3'd2;
    multi_detector_top #(.SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("../../../lut")) multi (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel), .detector_sel(detector_sel),
        .detect_valid(md_dv), .detect(md_d), .T_log_code(md_tlog)
    );

    reg [7:0] pixmem [0:N_PIX-1];
    integer i, fh, r3;
    integer errors, checked;

    initial begin
        fh = $fopen("../../../CFAR_Weibull/rtl/tb/top_pixels.txt", "r");
        for (i = 0; i < N_PIX; i = i + 1) r3 = $fscanf(fh, "%d\n", pixmem[i]);
        $fclose(fh);
    end

    initial begin
        errors = 0; checked = 0;
        rstn = 0; pixel_in_valid = 0; pixel_in = 0;
        repeat (5) @(posedge clk);
        rstn = 1;
        @(posedge clk);
        for (i = 0; i < N_PIX; i = i + 1) begin
            pixel_in_valid <= 1'b1;
            pixel_in       <= pixmem[i];
            @(posedge clk);
        end
        pixel_in_valid <= 1'b0;
        repeat (300) @(posedge clk);
        $display("checked=%0d errors=%0d", checked, errors);
        if (errors == 0) $display("PASS -- identical"); else $display("FAIL -- diverge");
        $finish;
    end

    integer wcount;
    initial wcount = -1;
    always @(posedge clk) begin
        if (gg_dv !== md_dv) begin
            $display("[%0t] dv MISMATCH: standalone=%b multi=%b", $time, gg_dv, md_dv);
            errors = errors + 1;
        end
        if (gg_dv) begin
            wcount = wcount + 1;
            checked = checked + 1;
            if (gg_d !== md_d || gg_tlog !== md_tlog) begin
                $display("[win %0d] detect: standalone=%b multi=%b   T_log: standalone=%0d multi=%0d",
                    wcount, gg_d, md_d, gg_tlog, md_tlog);
                errors = errors + 1;
            end
        end
    end
endmodule
