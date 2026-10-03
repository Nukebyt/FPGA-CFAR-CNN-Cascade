// peak5.v -- 5x5 contrast-peak event detector (streaming, raster input, one result per >= 5 clocks).
//
//   cd(y,x) = A for a gated detection, 0 otherwise  (A >= G > 0 for every gated pixel, so 0 acts as "-infinity")
//   event(y,x) = cd(y,x) != 0  &&  cd(y,x) >= max of cd over the 5x5 neighbourhood  (ties kept; outside the frame counts as 0)
//
// Separable maximum: hm(y,x) = max(cd(y, x-2..x+2)) from a 5-entry shift register; the vertical maximum over rows y-4..y comes
// from four column-indexed RAMs holding hm of the four previous rows (a shifting chain), which gives the centre row y-2.
// The centre pixel's cd and payload are delayed by two steps (2 columns, via the shift register) and two rows (RAM chain).
// After the last input pixel the module injects zero-valued steps itself (2 columns at every row end, then 2 extra rows),
// so the stream may stop at the last real pixel. Events leave in raster order.
`timescale 1ns/1ps
module peak5 #(
    parameter integer WP = 400,
    parameter integer HP = 400,
    parameter integer CW = 17,
    parameter integer PW = 44
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,            // frame start: clears the counters
    input  wire          in_valid,
    input  wire [CW-1:0] in_cd,
    input  wire [PW-1:0] in_pl,
    input  wire          in_last,          // with the last pixel of the frame
    output reg           ev_valid,
    output reg  [9:0]    ev_j,
    output reg  [9:0]    ev_i,
    output reg  [CW-1:0] ev_cd,
    output reg  [PW-1:0] ev_pl,
    output reg           done              // pulse once every event of the frame has been emitted
);
    // ---------------------------------------------------------------- step generator
    reg [11:0] x, y;                         // position of the step being issued
    reg        gen_on, gen_tog, final_f;
    reg [11:0] gen_left;                     // injected steps still to issue in the current burst
    reg        gen_rows;                     // 0: row-tail burst (2 steps), 1: the two flush rows
    reg [6:0]  dsr;
    wire       inj = gen_on & gen_tog;
    wire       step = in_valid | inj;
    wire [CW-1:0] cd_in = in_valid ? in_cd : {CW{1'b0}};
    wire [PW-1:0] pl_in = in_valid ? in_pl : {PW{1'b0}};

    // ---------------------------------------------------------------- stage 0 (step cycle)
    reg [CW-1:0] h1, h2, h3, h4;
    reg [PW-1:0] pl0, pl1;                   // payloads of the previous two columns
    wire xz = (x == 12'd0);
    wire [CW-1:0] w0 = xz ? {CW{1'b0}} : h1;
    wire [CW-1:0] w1 = xz ? {CW{1'b0}} : h2;
    wire [CW-1:0] w2 = xz ? {CW{1'b0}} : h3;
    wire [CW-1:0] w3 = xz ? {CW{1'b0}} : h4;
    wire [CW-1:0] w4 = cd_in;
    function [CW-1:0] mx; input [CW-1:0] a; input [CW-1:0] b; begin mx = (a > b) ? a : b; end endfunction
    wire [CW-1:0] pa_n  = mx(w0, w1);
    wire [CW-1:0] pb_n  = mx(w2, w3);
    wire [CW-1:0] cdc_n = w2;                                         // cd of column x-2
    wire [PW-1:0] plc_n = (x >= 12'd2) ? pl1 : {PW{1'b0}};            // payload of column x-2 (pl1 = x-2, pl0 = x-1 before this step)
    wire [11:0]   xc_n  = x - 12'd2;
    wire [11:0]   rd_n  = (x >= 12'd2) ? xc_n : 12'd0;

    // stage 0 (cycle after the step): pair maxima registered, final maximum computed here, RAM read address registered
    reg          s0_v;
    reg [11:0]   s0_x, s0_y, s0_rd;
    reg [CW-1:0] s0_pa, s0_pb, s0_w4, s0_cdc;
    reg [PW-1:0] s0_plc;
    wire [CW-1:0] hm_0 = mx(mx(s0_pa, s0_pb), s0_w4);
    wire [11:0]   rd_a = s0_rd;

    reg          s1_v;
    reg [11:0]   s1_x, s1_y;
    reg [CW-1:0] s1_hm, s1_cdc;
    reg [PW-1:0] s1_plc;
    always @(posedge clk) if (s0_v) begin s1_x <= s0_x; s1_y <= s0_y; s1_hm <= hm_0; s1_cdc <= s0_cdc; s1_plc <= s0_plc; end

    // column-indexed RAM chain
    localparam integer AW = (WP > 256) ? ((WP > 512) ? 10 : 9) : 8;
    reg [CW-1:0] hmr1 [0:WP-1];
    reg [CW-1:0] hmr2 [0:WP-1];
    reg [CW-1:0] hmr3 [0:WP-1];
    reg [CW-1:0] hmr4 [0:WP-1];
    reg [CW-1:0] cdr1 [0:WP-1];
    reg [CW-1:0] cdr2 [0:WP-1];
    reg [PW-1:0] plr1 [0:WP-1];
    reg [PW-1:0] plr2 [0:WP-1];
    reg [CW-1:0] q_hm1, q_hm2, q_hm3, q_hm4, q_cd1, q_cd2;
    reg [PW-1:0] q_pl1, q_pl2;
    always @(posedge clk) begin
        q_hm1 <= hmr1[rd_a]; q_hm2 <= hmr2[rd_a]; q_hm3 <= hmr3[rd_a]; q_hm4 <= hmr4[rd_a];
        q_cd1 <= cdr1[rd_a]; q_cd2 <= cdr2[rd_a];
        q_pl1 <= plr1[rd_a]; q_pl2 <= plr2[rd_a];
    end

    always @(posedge clk) begin
        if (!rstn || start) begin
            x <= 12'd0; y <= 12'd0; gen_on <= 1'b0; gen_tog <= 1'b0; final_f <= 1'b0; gen_left <= 12'd0; gen_rows <= 1'b0;
            s0_v <= 1'b0; s1_v <= 1'b0; done <= 1'b0; dsr <= 7'd0;
        end else begin
            gen_tog <= gen_on ? ~gen_tog : 1'b0;
            s0_v <= step & (x >= 12'd2); s1_v <= s0_v;
            if (step) begin
                h1 <= w1; h2 <= w2; h3 <= w3; h4 <= w4;
                pl0 <= pl_in; pl1 <= pl0;
                s0_x <= xc_n; s0_y <= y; s0_rd <= rd_n; s0_pa <= pa_n; s0_pb <= pb_n; s0_w4 <= w4; s0_cdc <= cdc_n; s0_plc <= plc_n;
                if (x == WP + 1) begin x <= 12'd0; y <= y + 12'd1; end else x <= x + 12'd1;
            end
            // burst control
            if (in_valid) begin
                if (x == WP - 1) begin gen_on <= 1'b1; gen_rows <= 1'b0; gen_left <= 12'd2; final_f <= in_last; gen_tog <= 1'b0; end
            end else if (inj) begin
                gen_left <= gen_left - 12'd1;
                if (gen_left == 12'd1) begin
                    if (!gen_rows && final_f) begin gen_rows <= 1'b1; gen_left <= 2 * (WP + 2); end
                    else gen_on <= 1'b0;
                end
            end
            dsr <= {dsr[5:0], (inj && gen_rows && gen_left == 12'd1)};
            done <= dsr[6];
        end
    end

    // ---------------------------------------------------------------- stage 1: RAM data in, RAM chain update, partial maxima
    wire row1 = (s1_y >= 12'd1), row2 = (s1_y >= 12'd2), row3 = (s1_y >= 12'd3), row4 = (s1_y >= 12'd4);
    wire [11:0] ycen = s1_y - 12'd2;
    reg          s2_v, s2_ok;
    reg [CW-1:0] s2_ma, s2_mb, s2_mc, s2_cd;
    reg [PW-1:0] s2_pl;
    reg [9:0]    s2_j, s2_i;
    always @(posedge clk) begin
        s2_v <= rstn & s1_v;
        if (rstn && s1_v) begin
            hmr1[s1_x] <= s1_hm; hmr2[s1_x] <= q_hm1; hmr3[s1_x] <= q_hm2; hmr4[s1_x] <= q_hm3;
            cdr1[s1_x] <= s1_cdc; cdr2[s1_x] <= q_cd1;
            plr1[s1_x] <= s1_plc; plr2[s1_x] <= q_pl1;
            s2_ma <= mx(s1_hm, row1 ? q_hm1 : {CW{1'b0}});
            s2_mb <= mx(row2 ? q_hm2 : {CW{1'b0}}, row3 ? q_hm3 : {CW{1'b0}});
            s2_mc <= row4 ? q_hm4 : {CW{1'b0}};
            s2_cd <= row2 ? q_cd2 : {CW{1'b0}};
            s2_pl <= q_pl2;
            s2_ok <= row2 && (ycen < HP);
            s2_j <= ycen[9:0]; s2_i <= s1_x[9:0];
        end
    end
    // ---------------------------------------------------------------- stage 2: vertical maximum
    reg          s3_v, s3_ok;
    reg [CW-1:0] s3_vmax, s3_cd;
    reg [PW-1:0] s3_pl;
    reg [9:0]    s3_j, s3_i;
    always @(posedge clk) begin
        s3_v <= rstn & s2_v;
        if (s2_v) begin
            s3_vmax <= mx(mx(s2_ma, s2_mb), s2_mc);
            s3_cd <= s2_cd; s3_pl <= s2_pl; s3_ok <= s2_ok; s3_j <= s2_j; s3_i <= s2_i;
        end
    end
    // ---------------------------------------------------------------- stage 3: event decision
    always @(posedge clk) begin
        ev_valid <= 1'b0;
        if (rstn && s3_v && s3_ok && s3_cd != {CW{1'b0}} && s3_cd >= s3_vmax) begin
            ev_valid <= 1'b1; ev_j <= s3_j; ev_i <= s3_i; ev_cd <= s3_cd; ev_pl <= s3_pl;
        end
    end
endmodule
