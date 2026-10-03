// gengamma_minimal_tb.v -- diagnostic only: rebuilds JUST the gengamma
// branch (front_end3 + gengamma_backend + c1/x delay + threshold_compare3),
// using the EXACT same wiring multi_detector_top.v uses, with NO other
// backends present at all. If this matches standalone gengamma_top exactly,
// the bug is specifically about co-existing with burr/g0 in the same
// design. If it does NOT match even here, the bug is in how the gengamma
// branch itself was rebuilt, independent of anything else.
`timescale 1ns/1ps
module gengamma_minimal_tb;
    localparam integer IMG_W = 48, IMG_H = 40;
    localparam integer N_PIX = IMG_W*IMG_H;

    reg clk = 0;
    reg rstn = 0;
    reg pixel_in_valid = 0;
    reg [7:0] pixel_in = 0;
    reg [1:0] pfa_sel = 1;

    always #5 clk = ~clk;

    // ---- standalone reference ----------------------------------------------
    wire s_dv, s_d; wire signed [16:0] s_tlog;
    gengamma_top #(.SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LUT_ROOT("../../../lut")) standalone (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel),
        .detect_valid(s_dv), .detect(s_d), .T_log_code(s_tlog)
    );

    // ---- minimal rebuild: own front_end3 + gengamma branch only -----------
    wire molc_valid;
    wire signed [15:0] c1_code;
    wire [25:0] c2_code;
    wire signed [27:0] c3_code;
    wire signed [15:0] x_code;
    front_end3 #(.SLI(17), .GUARD(13), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LOG_AMP_HEX("../../../lut/shared/log_amp_lut.hex")) fe (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .molc_valid(molc_valid), .c1_code(c1_code), .c2_code(c2_code),
        .c3_code(c3_code), .x_code(x_code)
    );

    wire gg_valid; wire signed [15:0] gg_delta;
    gengamma_backend #(.LUT_ROOT("../../../lut")) gg_be (
        .clk(clk), .rstn(rstn), .molc_valid(molc_valid),
        .c2_code(c2_code), .c3_code(c3_code), .pfa_sel(pfa_sel),
        .valid_out(gg_valid), .delta_out(gg_delta),
        .addr_sat_low(), .addr_sat_high()
    );
    wire gg_c1v; wire signed [15:0] gg_c1d;
    wire gg_xv;  wire signed [15:0] gg_xd;
    pipe_delay #(.DATA_WIDTH(16), .DEPTH(7)) gg_c1_delay (.clk(clk), .rstn(rstn), .in_valid(molc_valid), .in_data(c1_code), .out_valid(gg_c1v), .out_data(gg_c1d));
    pipe_delay #(.DATA_WIDTH(16), .DEPTH(7)) gg_x_delay  (.clk(clk), .rstn(rstn), .in_valid(molc_valid), .in_data(x_code),  .out_valid(gg_xv),  .out_data(gg_xd));
    wire gg_valid_gated = gg_valid & gg_c1v;
    wire m_dv, m_d; wire signed [16:0] m_tlog;
    threshold_compare3 gg_tc (.clk(clk), .rstn(rstn), .in_valid(gg_valid_gated), .c1_code(gg_c1d), .delta_code(gg_delta), .x_code(gg_xd), .detect_valid(m_dv), .detect(m_d), .T_log_code(m_tlog));

    reg [7:0] pixmem [0:N_PIX-1];
    integer i, fh, r3;
    integer errors, checked, wcount;

    initial begin
        fh = $fopen("../../../CFAR_Weibull/rtl/tb/top_pixels.txt", "r");
        for (i = 0; i < N_PIX; i = i + 1) r3 = $fscanf(fh, "%d\n", pixmem[i]);
        $fclose(fh);
    end

    initial begin
        errors = 0; checked = 0; wcount = -1;
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
        if (errors == 0) $display("PASS -- minimal rebuild matches standalone exactly");
        else $display("FAIL -- diverges even in minimal rebuild");
        $finish;
    end

    always @(posedge clk) begin
        if (s_dv !== m_dv) begin
            $display("[%0t] dv MISMATCH: standalone=%b minimal=%b", $time, s_dv, m_dv);
            errors = errors + 1;
        end
        if (s_dv) begin
            wcount = wcount + 1;
            checked = checked + 1;
            if (s_d !== m_d || s_tlog !== m_tlog) begin
                $display("[win %0d] detect: standalone=%b minimal=%b   T_log: standalone=%0d minimal=%0d",
                    wcount, s_d, m_d, s_tlog, m_tlog);
                errors = errors + 1;
            end
        end
    end
endmodule
