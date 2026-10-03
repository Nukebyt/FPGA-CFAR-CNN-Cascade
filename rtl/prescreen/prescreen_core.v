// prescreen_core.v -- pooled-domain Weibull prescreen, pass 2 of the cascade.
//
// Reads the 2x2-pooled 8-bit frame P (WP x HP, raster) back out of the pooled store with MIRRORED border addresses
// (numpy 'symmetric': edge sample repeated) and, for every position of the padded frame, derives the ring statistics of
// an SLI x SLI window minus the GUARD x GUARD window with running column sums:
//
//   per padded pixel (r,c) five store reads (one per clock, slot 0..4):
//     0: Pp(r, c)               1: Pp(r-SLI, c)           2: Pp(r-(TK-TG), c)
//     3: Pp(r-(TK+TG+1), c)     4: Pp(r-TK, c-TK)  (the centre pixel)
//   V25[c] += Pp(r,c)-Pp(r-SLI,c)            (vertical sum over the SLI rows r-SLI+1..r, for x and x^2)
//   V17[c] += Pp(r-TK+TG,c)-Pp(r-TK-TG-1,c)  (vertical sum over the guard rows)
//   H25 / H17: horizontal running sums of V25 / V17 (shift registers of the new column sums)
//   S1 = H25x-H17x ; S2 = H25xx-H17xx          ring sums of P and P^2 (N = SLI^2-GUARD^2 cells)
//
// then, bit-exactly as _comparison/fixedpoint/prescreen_fx.py:
//   A   = N*P - S1                              contrast x N
//   num = N*S2 - S1^2                           = N(N-1) c2
//   D   = (A > 0) && (A^2 > (clamp(num,NUM_LO,NUM_HI) >> 12) * KC[pfa_sel])        Weibull decision, no ROM / sqrt / divider
//   Dg  = D && (A >= G)                         gate
// The stream (valid every 5th clock) of  cd = Dg ? A : 0  and a payload {S1, num>>12, P} goes to peak5.v.
//
// Throughput: (HP+2TK)*(WP+2TK)*5 clocks per frame (config A, 400x400 pooled: 0.90 M clocks).
`timescale 1ns/1ps
module prescreen_core #(
    parameter integer WP = 400,
    parameter integer HP = 400,
    parameter integer SLI = 25,
    parameter integer GUARD = 17,
    parameter integer AW = 18,
    parameter integer KC0 = 8381,      // Kc*2^12 for Pfa 3e-2, 1e-2, 1e-3, 1e-4 (sli 25 / guard 17, N = 336)
    parameter integer KC1 = 11060,
    parameter integer KC2 = 15733,
    parameter integer KC3 = 19546,
    parameter integer NUM_LO = 18371009,
    parameter integer NUM_HI = 1837100899
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,                // 1-clock pulse (while idle)
    input  wire [1:0]    pfa_sel,
    input  wire [16:0]   g_th,                 // gate on A (N * tau / step)
    output reg  [AW-1:0] ps_raddr,             // pooled-store read port (synchronous read: data one clock after the address register)
    input  wire [7:0]    ps_rdata,
    output reg           busy,
    // result stream (one result every 5th clock while running)
    output reg           o_valid,
    output reg  [16:0]   o_cd,                 // gated contrast (A) or 0
    output reg  [43:0]   o_pl,                 // {S1[16:0], numsh[18:0], P[7:0]}
    output reg           o_last                // last result of the frame
);
    localparam integer TK = (SLI - 1) / 2;
    localparam integer TG = (GUARD - 1) / 2;
    localparam integer N  = SLI * SLI - GUARD * GUARD;
    localparam integer WR = WP + 2 * TK;
    localparam integer HR = HP + 2 * TK;
    localparam integer EN17 = TK - TG;          // guard window enters EN17 columns/rows behind the newest
    localparam integer LV17 = TK + TG + 1;      // ... and leaves LV17 behind

    function integer clog2; input integer v; integer kk; begin clog2 = 0; for (kk = v - 1; kk > 0; kk = kk >> 1) clog2 = clog2 + 1; end endfunction
    localparam integer CW = clog2(WR);

    // ------------------------------------------------------------------ sequencer: slot s (0..4), padded row r, padded column c
    reg [2:0] s;
    reg [11:0] r, c;
    reg last_tok;

    function [11:0] mir;
        input integer i; input integer n;
        integer t;
        begin
            t = i - TK;
            if (t < 0) mir = -t - 1; else if (t >= n) mir = 2 * n - 1 - t; else mir = t;
        end
    endfunction

    integer ri, ci;
    always @* begin
        case (s)
            3'd0: begin ri = r;                ci = c;      end
            3'd1: begin ri = r - SLI;          ci = c;      end
            3'd2: begin ri = r - EN17;         ci = c;      end
            3'd3: begin ri = r - LV17;         ci = c;      end
            default: begin ri = r - TK;        ci = c - TK; end
        endcase
    end

    reg [11:0] mrow, mcol;
    reg        sv1, sv2, sv3;
    reg [2:0]  sd1, sd2, sd3;
    reg [11:0] rq1, rq2, rq3, cq1, cq2, cq3;
    reg        lq1, lq2, lq3;

    always @(posedge clk) begin
        if (!rstn) begin
            s <= 3'd0; r <= 12'd0; c <= 12'd0; busy <= 1'b0; last_tok <= 1'b0;
            sv1 <= 1'b0; sv2 <= 1'b0; sv3 <= 1'b0;
        end else begin
            sv1 <= busy; sv2 <= sv1; sv3 <= sv2;
            if (start && !busy) begin busy <= 1'b1; s <= 3'd0; r <= 12'd0; c <= 12'd0; last_tok <= 1'b0; end
            else if (busy) begin
                if (s == 3'd4) begin
                    s <= 3'd0;
                    if (c == WR - 1) begin
                        c <= 12'd0;
                        if (r == HR - 1) busy <= 1'b0; else r <= r + 1'b1;
                    end else c <= c + 1'b1;
                end else s <= s + 1'b1;
                last_tok <= (c == WR - 1) && (r == HR - 1);
            end
        end
        sd1 <= s; sd2 <= sd1; sd3 <= sd2;
        rq1 <= r; rq2 <= rq1; rq3 <= rq2; cq1 <= c; cq2 <= cq1; cq3 <= cq2;
        lq1 <= last_tok; lq2 <= lq1; lq3 <= lq2;
        mrow <= mir(ri, HP);
        mcol <= mir(ci, WP);
        ps_raddr <= mrow * WP + mcol;
    end

    // ------------------------------------------------------------------ capture of the five reads
    reg [7:0] p0, p1, p2, p3, p4;
    reg       tok_go;                       // all five reads of a token are in p0..p4 (this cycle)
    reg [11:0] r_u, c_u;
    reg        l_u;
    always @(posedge clk) begin
        tok_go <= 1'b0;
        if (rstn && sv3) begin
            case (sd3)
                3'd0: p0 <= ps_rdata;
                3'd1: p1 <= ps_rdata;
                3'd2: p2 <= ps_rdata;
                3'd3: p3 <= ps_rdata;
                default: begin p4 <= ps_rdata; tok_go <= 1'b1; r_u <= rq3; c_u <= cq3; l_u <= lq3; end
            endcase
        end
    end

    // ------------------------------------------------------------------ stage U: deltas of the vertical sums
    wire m1 = (r_u >= SLI);
    wire m2 = (r_u >= EN17);
    wire m3 = (r_u >= LV17);
    wire [7:0] a_in25  = p0;
    wire [7:0] a_old25 = m1 ? p1 : 8'd0;
    wire [7:0] a_in17  = m2 ? p2 : 8'd0;
    wire [7:0] a_old17 = m3 ? p3 : 8'd0;
    wire [15:0] sq_in25  = a_in25 * a_in25;
    wire [15:0] sq_old25 = a_old25 * a_old25;
    wire [15:0] sq_in17  = a_in17 * a_in17;
    wire [15:0] sq_old17 = a_old17 * a_old17;
    reg signed [8:0]  dx25, dx17;
    reg signed [16:0] dxx25, dxx17;
    reg        v_v, l_v;
    reg [11:0] r_v, c_v;
    reg [7:0]  p4_v;
    always @(posedge clk) begin
        v_v <= rstn & tok_go;
        if (tok_go) begin
            dx25  <= $signed({1'b0, a_in25})  - $signed({1'b0, a_old25});
            dx17  <= $signed({1'b0, a_in17})  - $signed({1'b0, a_old17});
            dxx25 <= $signed({1'b0, sq_in25}) - $signed({1'b0, sq_old25});
            dxx17 <= $signed({1'b0, sq_in17}) - $signed({1'b0, sq_old17});
            r_v <= r_u; c_v <= c_u; p4_v <= p4; l_v <= l_u;
        end
    end

    // column-sum RAMs (read at c_u during stage U, data in stage V; written back in stage V)
    reg [12:0] v25x [0:WR-1];
    reg [20:0] v25xx [0:WR-1];
    reg [12:0] v17x [0:WR-1];
    reg [20:0] v17xx [0:WR-1];
    reg [12:0] q25x, q17x;
    reg [20:0] q25xx, q17xx;
    always @(posedge clk) begin
        q25x <= v25x[c_u[CW-1:0]]; q25xx <= v25xx[c_u[CW-1:0]];
        q17x <= v17x[c_u[CW-1:0]]; q17xx <= v17xx[c_u[CW-1:0]];
    end

    // ------------------------------------------------------------------ stage V: new column sums
    wire first_row = (r_v == 12'd0);
    wire [12:0] n25x  = (first_row ? 13'd0 : q25x)  + {{4{dx25[8]}}, dx25};
    wire [20:0] n25xx = (first_row ? 21'd0 : q25xx) + {{4{dxx25[16]}}, dxx25};
    wire [12:0] n17x  = (first_row ? 13'd0 : q17x)  + {{4{dx17[8]}}, dx17};
    wire [20:0] n17xx = (first_row ? 21'd0 : q17xx) + {{4{dxx17[16]}}, dxx17};
    always @(posedge clk) if (v_v) begin
        v25x[c_v[CW-1:0]] <= n25x; v25xx[c_v[CW-1:0]] <= n25xx;
        v17x[c_v[CW-1:0]] <= n17x; v17xx[c_v[CW-1:0]] <= n17xx;
    end
    reg        h_v, h_l;
    reg [11:0] h_r, h_c;
    reg [7:0]  h_p;
    reg [12:0] h_n25x, h_n17x;
    reg [20:0] h_n25xx, h_n17xx;
    always @(posedge clk) begin
        h_v <= rstn & v_v;
        if (v_v) begin
            h_n25x <= n25x; h_n25xx <= n25xx; h_n17x <= n17x; h_n17xx <= n17xx;
            h_r <= r_v; h_c <= c_v; h_p <= p4_v; h_l <= l_v;
        end
    end

    // ------------------------------------------------------------------ stage H: horizontal running sums
    reg [12:0] sr25x [0:SLI-1];  reg [20:0] sr25xx [0:SLI-1];
    reg [12:0] sr17x [0:LV17-1]; reg [20:0] sr17xx [0:LV17-1];
    reg [17:0] H25x, H17x;       reg [25:0] H25xx, H17xx;
    integer k;
    wire c0 = (h_c == 12'd0);
    localparam integer EI = (EN17 == 0) ? 0 : EN17 - 1;
    wire [12:0] lv25x  = (h_c >= SLI)  ? sr25x[SLI-1]   : 13'd0;
    wire [20:0] lv25xx = (h_c >= SLI)  ? sr25xx[SLI-1]  : 21'd0;
    wire [12:0] en17x  = (h_c >= EN17) ? ((EN17 == 0) ? h_n17x  : sr17x[EI])  : 13'd0;
    wire [20:0] en17xx = (h_c >= EN17) ? ((EN17 == 0) ? h_n17xx : sr17xx[EI]) : 21'd0;
    wire [12:0] lv17x  = (h_c >= LV17) ? sr17x[LV17-1]  : 13'd0;
    wire [20:0] lv17xx = (h_c >= LV17) ? sr17xx[LV17-1] : 21'd0;
    wire [17:0] nH25x  = (c0 ? 18'd0 : H25x)  + h_n25x  - lv25x;
    wire [25:0] nH25xx = (c0 ? 26'd0 : H25xx) + h_n25xx - lv25xx;
    wire [17:0] nH17x  = (c0 ? 18'd0 : H17x)  + en17x   - lv17x;
    wire [25:0] nH17xx = (c0 ? 26'd0 : H17xx) + en17xx  - lv17xx;
    reg        s_v, s_l;
    reg [16:0] s1_o;
    reg [24:0] s2_o;
    reg [7:0]  p_o;
    always @(posedge clk) begin
        s_v <= 1'b0;
        if (h_v) begin
            H25x <= nH25x; H25xx <= nH25xx; H17x <= nH17x; H17xx <= nH17xx;
            sr25x[0] <= h_n25x; sr25xx[0] <= h_n25xx; sr17x[0] <= h_n17x; sr17xx[0] <= h_n17xx;
            for (k = 1; k < SLI;  k = k + 1) begin sr25x[k] <= sr25x[k-1];  sr25xx[k] <= sr25xx[k-1];  end
            for (k = 1; k < LV17; k = k + 1) begin sr17x[k] <= sr17x[k-1];  sr17xx[k] <= sr17xx[k-1];  end
            s_v <= rstn && (h_r >= 2 * TK) && (h_c >= 2 * TK);
            s1_o <= nH25x - nH17x; s2_o <= nH25xx - nH17xx; p_o <= h_p; s_l <= h_l;
        end
    end

    // ------------------------------------------------------------------ arithmetic
    // a1: A, N*S2, S1^2
    reg        a1_v, a1_l;
    reg signed [18:0] a1_A;
    reg [33:0] a1_ns2, a1_s1sq;
    reg [16:0] a1_s1;
    reg [7:0]  a1_p;
    wire [33:0] ns2_w  = N * s2_o;
    wire [33:0] s1sq_w = s1_o * s1_o;
    wire signed [18:0] a_w = $signed(N * p_o) - $signed({2'b0, s1_o});
    always @(posedge clk) begin
        a1_v <= s_v; a1_l <= s_l;
        a1_A    <= a_w;
        a1_ns2  <= ns2_w;
        a1_s1sq <= s1sq_w;
        a1_s1 <= s1_o; a1_p <= p_o;
    end
    // a2: num, A^2
    reg        a2_v, a2_l;
    reg signed [18:0] a2_A;
    reg [33:0] a2_num, a2_asq;
    reg [16:0] a2_s1; reg [7:0] a2_p;
    wire [35:0] asq_w = a1_A[17:0] * a1_A[17:0];
    always @(posedge clk) begin
        a2_v <= a1_v; a2_l <= a1_l;
        a2_A   <= a1_A;
        a2_num <= a1_ns2 - a1_s1sq;
        a2_asq <= (a1_A > 0) ? asq_w[33:0] : 34'd0;
        a2_s1 <= a1_s1; a2_p <= a1_p;
    end
    // a3: clamp and shift
    reg        a3_v, a3_l;
    reg signed [18:0] a3_A;
    reg [33:0] a3_asq;
    reg [18:0] a3_nsh;
    reg [16:0] a3_s1; reg [7:0] a3_p;
    wire [33:0] numc = (a2_num < NUM_LO) ? NUM_LO : ((a2_num > NUM_HI) ? NUM_HI : a2_num);
    always @(posedge clk) begin
        a3_v <= a2_v; a3_l <= a2_l; a3_A <= a2_A; a3_asq <= a2_asq;
        a3_nsh <= numc[30:12];
        a3_s1 <= a2_s1; a3_p <= a2_p;
    end
    // a4: Kc * (num >> 12)
    reg [15:0] kc;
    always @(posedge clk) kc <= (pfa_sel == 2'd0) ? KC0 : (pfa_sel == 2'd1) ? KC1 : (pfa_sel == 2'd2) ? KC2 : KC3;
    reg        a4_v, a4_l;
    reg signed [18:0] a4_A;
    reg [33:0] a4_asq, a4_R;
    reg [18:0] a4_nsh; reg [16:0] a4_s1; reg [7:0] a4_p;
    wire [34:0] r_w = a3_nsh * kc;
    always @(posedge clk) begin
        a4_v <= a3_v; a4_l <= a3_l; a4_A <= a3_A; a4_asq <= a3_asq;
        a4_R <= r_w[33:0];
        a4_nsh <= a3_nsh; a4_s1 <= a3_s1; a4_p <= a3_p;
    end
    // a5: decision + gate
    always @(posedge clk) begin
        if (!rstn) begin o_valid <= 1'b0; o_last <= 1'b0; end
        else begin
            o_valid <= a4_v; o_last <= a4_v & a4_l;
            o_cd <= ((a4_A > 0) && (a4_asq > a4_R) && ({1'b0, a4_A[17:0]} >= {1'b0, g_th})) ? a4_A[16:0] : 17'd0;
            o_pl <= {a4_s1, a4_nsh, a4_p};
        end
    end
endmodule
