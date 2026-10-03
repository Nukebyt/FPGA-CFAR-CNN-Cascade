// cnn_core_ctx.v -- cnn_core_q4 generalised to the context cascade network (see gen_cnn_rtl_ctx.py): THREE input streams per candidate
// (fine patch, context patch, 9 side bytes; `in_kind` 0/1/2, any order), per-layer source/destination memory and base offsets, and layers whose
// accumulation runs over up to three input SEGMENTS (concatenated head). Arithmetic is unchanged and bit-exact to the Python integer model.
// (original header follows)
// cnn_core_q4.v -- INT8 CNN discriminator, quad-pixel MAC array (4x the throughput of cnn_core.v, same interface,
// same arithmetic, bit-exact to cnn/quant_hw.py int_forward).
//
// Throughput idea: every pass computes a 2x2 QUAD of output pixels (lanes x 4 pixels = 4*L MACs per cycle) from
// ONE weight word per cycle. To read the four input pixels (oy+ky+dy, ox+kx+dx) in the same cycle, every feature
// map is stored in 4 PARITY BANKS:
//      bank(y,x) = 2*(y&1) + (x&1),     addr = c*BSZ + (y>>1)*WB + (x>>1)
// For a quad whose origin is even/even, tap (ky,kx) hits four DIFFERENT banks; which pixel each bank serves is a
// permutation selected by (ky&1, kx&1) -- a 4x4 byte crossbar after the RAM read.
//   * POOL layers: the quad is exactly the 2x2 pool window. The max is taken on the raw accumulators (requant is
//     monotone => max-then-requant == requant-then-max), so each lane needs ONE requant per quad, not four.
//   * QUAD layers (conv, no pool): four requants per lane per quad, written to the four banks (same address).
//   * ONE layers (FC, 1x1 output; FC-after-conv is a K=H conv): only pixel lane 0 is used.
// Layer table / banked geometry come from gen_cnn_rtl_q4.py (cnn_cfg.vh); weight/bias images are the same as the
// serial core's. Multiplier style: pixel lanes < LOGIC_FROM use DSP blocks, the rest are built in ALMs
// (one DSP block per 9x9 multiplier would exhaust the device at 4*L lanes).
`timescale 1ns/1ps

(* multstyle = "dsp" *)
module cnn_mul_dsp (input wire clk, input wire signed [8:0] a, input wire signed [7:0] w, output reg signed [17:0] p);
    always @(posedge clk) p <= a * w;
endmodule

(* multstyle = "logic" *)
module cnn_mul_logic (input wire clk, input wire signed [8:0] a, input wire signed [7:0] w, output reg signed [17:0] p);
    always @(posedge clk) p <= a * w;
endmodule

module cnn_core_ctx #(
    parameter W_HEX  = "cnn_w.hex",
    parameter PQ_HEX = "cnn_pq.hex",
    parameter integer ACCW = 26,
    parameter integer LOGIC_FROM = 2           // pixel lanes p >= LOGIC_FROM: multipliers in ALMs
) (
    input  wire               clk,
    input  wire               rstn,
    input  wire               in_valid,
    input  wire [7:0]         in_data,
    input  wire [1:0]         in_kind,          // 0: fine patch pixel, 1: context patch pixel, 2: side byte
    output wire               ready,
    output reg                done,
    output reg  signed [31:0] logit,
    input  wire signed [31:0] theta,
    output reg                detect
);
`include "cnn_cfg.vh"
    localparam integer L = LANES_GEN;
    localparam [1:0] M_ONE = 2'd0, M_QUAD = 2'd1, M_POOL = 2'd2;

    // ------------------------------------------------------------------ state
    localparam [3:0] S_IDLE = 4'd0, S_LOAD = 4'd1, S_LCFG = 4'd2, S_GRP = 4'd3,
                     S_PIX = 4'd4, S_DRAIN = 4'd5, S_DONE = 4'd6;
    reg [3:0] st;
    assign ready = (st == S_IDLE);

    reg [6:0]     ld_x, ld_y;
    reg [AW-1:0]  ld_row;
    reg [3:0]     lyr;
    reg [3:0]     g;
    reg [9:0]     gL;
    reg [AW-1:0]  gch;
    reg [WAW-1:0] wgrp;
    reg [PQAW-1:0] pqgrp;
    reg [6:0]     Lv;
    reg [2:0]     got;                  // which input streams of this candidate are complete
    reg [3:0]     side_cnt;
    reg [1:0]     sg;                   // current input segment of the layer
    reg [9:0]     cch;                  // channel counter inside the segment
    reg [9:0]     c_SCIN;
    reg           c_IMEM, c_OMEM;
    reg [AW-1:0]  c_IBASE, c_OBASE, c_WBL;
    reg [1:0]     c_NSEG;

    // layer configuration registers
    reg [3:0]     c_K;
    reg [TW-1:0]  c_T;
    reg [1:0]     c_MODE;
    reg [5:0]     c_QH, c_QW;
    reg [AW-1:0]  c_WBIN, c_BSZIN, c_WBO, c_BSZO;
    reg           c_relu;
    reg [5:0]     c_S;
    reg [3:0]     c_G;
    reg [9:0]     c_Cout;

    // issue-engine state
    reg [5:0]     qy, qx;
    reg [AW-1:0]  base_pix, cioff, orow;
    reg [TW-1:0]  tcnt;
    reg [2:0]     kx, ky;
    reg [WAW-1:0] wptr;

    // ---------------------------------------------------------- pipeline regs
    // i0: partial addresses; iss: bank addresses (RAM read issued); t1: data valid; t2: product valid
    reg           i0_v, i0_first, i0_last, i0_ky0, i0_kx0;
    reg [AW-1:0]  i0_sr0, i0_sr1, i0_opix;
    reg [2:0]     i0_c0, i0_c1;
    reg [1:0]     i0_obank;
    reg [WAW-1:0] i0_wa;
    reg           iss_v, iss_first, iss_last, iss_ky0, iss_kx0;
    reg [AW-1:0]  iss_opix;
    reg [1:0]     iss_obank;
    reg [AW-1:0]  a_addr [0:3];
    reg [WAW-1:0] wa_addr;
    reg           t1_v, t1_first, t1_last, t1_ky0, t1_kx0;
    reg [AW-1:0]  t1_opix;
    reg [1:0]     t1_obank;
    reg           t1b_v, t1b_first, t1b_last;     // crossbar output / multiplier input registers
    reg [AW-1:0]  t1b_opix;
    reg [1:0]     t1b_obank;
    reg           t2_v, t2_first, t2_last;
    reg [AW-1:0]  t2_opix;
    reg [1:0]     t2_obank;

    // ------------------------------------------------------------- memories
    reg           rdsel;
    wire [3:0]    wA_en, wB_en;
    wire [AW-1:0] w_addr_o;
    wire [7:0]    w_data_o;
    wire [31:0]   rd_all;
    genvar gb;
    generate
        for (gb = 0; gb < 4; gb = gb + 1) begin : bank
            reg [7:0] mA [0:BANK_DEPTH-1];
            reg [7:0] mB [0:BANK_DEPTH-1];
            reg [7:0] rA, rB;
            always @(posedge clk) begin
                if (wA_en[gb]) mA[w_addr_o] <= w_data_o;
                rA <= mA[a_addr[gb]];
            end
            always @(posedge clk) begin
                if (wB_en[gb]) mB[w_addr_o] <= w_data_o;
                rB <= mB[a_addr[gb]];
            end
            assign rd_all[8*gb +: 8] = rdsel ? rB : rA;
        end
    endgenerate

    reg [L*8-1:0] wmem [0:WDEPTH-1];
    reg [47:0]    pqmem [0:PQDEPTH-1];
    initial begin
        $readmemh(W_HEX, wmem);
        $readmemh(PQ_HEX, pqmem);
    end
    reg [L*8-1:0] w_q, w_q2;
    reg [47:0]    pq_q;
    reg [PQAW-1:0] pq_addr;
    always @(posedge clk) begin
        w_q  <= wmem[wa_addr];
        w_q2 <= w_q;
        pq_q <= pqmem[pq_addr];
    end

    // patch loader (banked) and requant writer share one write address/data bus
    wire ld_we = in_valid && (st == S_IDLE || st == S_LOAD);
    wire [1:0]    ld_bank = (in_kind == 2'd2) ? 2'd0 : {ld_y[0], ld_x[0]};
    wire [AW-1:0] ld_addr = (in_kind == 2'd2) ? (SIDE_BASE + side_cnt) : (ld_row + (ld_x >> 1) + ((in_kind == 2'd1) ? CTX_BASE : 0));
    reg  [3:0]    rq_we;
    reg  [AW-1:0] rq_waddr;
    reg  [7:0]    rq_wdata;
    assign wA_en = ({4{ld_we}} & (4'd1 << ld_bank)) | ({4{~c_OMEM}} & rq_we);
    assign wB_en = {4{c_OMEM}} & rq_we;
    assign w_addr_o = ld_we ? ld_addr : rq_waddr;
    assign w_data_o = ld_we ? in_data : rq_wdata;

    // ----------------------------------------------------- MAC array (L x 4)
    reg signed [ACCW-1:0] acc [0:4*L-1];
    reg signed [ACCW-1:0] sh  [0:4*L-1];
    genvar gl, gp;
    generate
        for (gp = 0; gp < 4; gp = gp + 1) begin : pix
            // pixel p = {dy,dx} reads bank {dy^ky0, dx^kx0}
            localparam [1:0] PB = gp;
            wire [1:0] bsel = PB ^ {t1_ky0, t1_kx0};
            reg  [7:0] act;
            always @(posedge clk) act <= rd_all[8*bsel +: 8];       // 4x4 byte crossbar, registered
            wire signed [8:0] a9 = {1'b0, act};
            for (gl = 0; gl < L; gl = gl + 1) begin : lane
                wire signed [7:0] w8 = w_q2[8*gl +: 8];
                wire signed [17:0] prod;
                if (gp < LOGIC_FROM) begin : u
                    cnn_mul_dsp m (.clk(clk), .a(a9), .w(w8), .p(prod));
                end else begin : u
                    cnn_mul_logic m (.clk(clk), .a(a9), .w(w8), .p(prod));
                end
                wire signed [ACCW-1:0] nxt = t2_first ? $signed(prod) : (acc[gl*4+gp] + $signed(prod));
                always @(posedge clk) begin
                    if (t2_v) begin
                        acc[gl*4+gp] <= nxt;
                        if (t2_last) sh[gl*4+gp] <= nxt;
                    end
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------ requant walk
    reg           rq_busy, rq_issue;
    reg [6:0]     l_iss;
    reg [1:0]     p_iss;
    reg [1:0]     rq_obank;
    reg [AW-1:0]  rq_obase, oa;
    reg           ra_v, ra_last, ra_pool;       // stage A: the four candidate accumulators of the current lane
    reg [AW-1:0]  ra_addr;
    reg [1:0]     ra_bank, ra_p;
    reg signed [ACCW-1:0] ra_s0, ra_s1, ra_s2, ra_s3;
    reg           r0b_v, r0b_last;
    reg [AW-1:0]  r0b_addr;
    reg [1:0]     r0b_bank;
    reg signed [ACCW-1:0] r0b_acc;
    reg           r1_v, r1_last;
    reg [AW-1:0]  r1_addr;
    reg [1:0]     r1_bank;
    reg signed [ACCW-1:0] r1_tot;
    reg [15:0]    r1_m;
    reg           r2_v, r2_last;
    reg [AW-1:0]  r2_addr;
    reg [1:0]     r2_bank;
    reg signed [ACCW-1:0] r2_tot;
    reg signed [ACCW+16:0] r2_prod;
    reg           r3_v, r3_last;
    reg [AW-1:0]  r3_addr;
    reg [1:0]     r3_bank;
    reg signed [ACCW-1:0] r3_tot;
    reg [7:0]     r3_y;

    wire signed [ACCW+16:0] rnd = (c_S == 0) ? {(ACCW+17){1'b0}} : ({{(ACCW+16){1'b0}}, 1'b1} <<< (c_S - 1));
    wire signed [ACCW+16:0] r2_rnd = r2_prod + rnd;
    wire signed [ACCW+16:0] r2_shr = r2_rnd >>> c_S;
    wire [7:0] y_clamped = (r2_shr < 0) ? 8'd0 : (r2_shr > 255) ? 8'd255 : r2_shr[7:0];

    wire [1:0] n_p_m1 = (c_MODE == M_QUAD) ? 2'd3 : 2'd0;          // pixels per lane in the walk, minus 1
    wire signed [ACCW-1:0] mx01 = (ra_s0 > ra_s1) ? ra_s0 : ra_s1;      // stage B (from the registered candidates)
    wire signed [ACCW-1:0] mx23 = (ra_s2 > ra_s3) ? ra_s2 : ra_s3;
    wire signed [ACCW-1:0] mx   = (mx01 > mx23) ? mx01 : mx23;
    wire signed [ACCW-1:0] shp  = (ra_p == 2'd0) ? ra_s0 : (ra_p == 2'd1) ? ra_s1 : (ra_p == 2'd2) ? ra_s2 : ra_s3;
    wire signed [ACCW-1:0] walk_val = ra_pool ? mx : shp;
    wire [1:0] walk_bank_i = (c_MODE == M_QUAD) ? p_iss : (c_MODE == M_POOL) ? rq_obank : 2'd0;

    // ------------------------------------------------------------ issue control
    wire qx_last = (qx == c_QW - 1);
    wire qy_last = (qy == c_QH - 1);
    wire final_quad = qx_last && qy_last;
    wire last_tap  = (tcnt == c_T - 1);
    wire last_inflight = (i0_v & i0_last) | (iss_v & iss_last) | (t1_v & t1_last) | (t1b_v & t1b_last) | (t2_v & t2_last);
    wire stall = last_tap & (rq_busy | last_inflight);
    wire pipe_empty = ~i0_v & ~iss_v & ~t1_v & ~t1b_v & ~t2_v & ~rq_busy;

    wire [2:0] r0n = (ky + 3'd1) >> 1;
    wire [2:0] r1n = ky >> 1;
    wire [2:0] c0n = (kx + 3'd1) >> 1;
    wire [2:0] c1n = kx >> 1;
    function [AW-1:0] rowmul;
        input [2:0] r; input [AW-1:0] wb;
        case (r)
            3'd0: rowmul = {AW{1'b0}};
            3'd1: rowmul = wb;
            3'd2: rowmul = wb << 1;
            default: rowmul = (wb << 1) + wb;
        endcase
    endfunction
    wire [AW-1:0] opix_now = orow + ((c_MODE == M_POOL) ? (qx >> 1) : qx);
    wire [1:0]    obank_now = {qy[0], qx[0]};

    always @(posedge clk) begin
        if (!rstn) begin
            st <= S_IDLE; done <= 1'b0; detect <= 1'b0; logit <= 32'sd0;
            i0_v <= 1'b0; iss_v <= 1'b0; t1_v <= 1'b0; t1b_v <= 1'b0; t2_v <= 1'b0;
            rq_busy <= 1'b0; rq_issue <= 1'b0; ra_v <= 1'b0; r0b_v <= 1'b0; r1_v <= 1'b0; r2_v <= 1'b0; r3_v <= 1'b0;
            rq_we <= 4'd0; ld_x <= 7'd0; ld_y <= 7'd0; ld_row <= {AW{1'b0}}; lyr <= 4'd0; got <= 3'd0; side_cnt <= 4'd0;
        end else begin
            done  <= 1'b0;
            rq_we <= 4'd0;
            // ---- pipeline advance
            iss_v <= i0_v; iss_first <= i0_first; iss_last <= i0_last; iss_ky0 <= i0_ky0; iss_kx0 <= i0_kx0;
            iss_opix <= i0_opix; iss_obank <= i0_obank;
            a_addr[0] <= i0_sr0 + i0_c0; a_addr[1] <= i0_sr0 + i0_c1;
            a_addr[2] <= i0_sr1 + i0_c0; a_addr[3] <= i0_sr1 + i0_c1;
            wa_addr <= i0_wa;
            t1_v <= iss_v; t1_first <= iss_first; t1_last <= iss_last; t1_ky0 <= iss_ky0; t1_kx0 <= iss_kx0;
            t1_opix <= iss_opix; t1_obank <= iss_obank;
            t1b_v <= t1_v; t1b_first <= t1_first; t1b_last <= t1_last; t1b_opix <= t1_opix; t1b_obank <= t1_obank;
            t2_v <= t1b_v; t2_first <= t1b_first; t2_last <= t1b_last; t2_opix <= t1b_opix; t2_obank <= t1b_obank;
            i0_v <= 1'b0;

            // ---- capture: last tap of a quad accumulated -> start the requant walk
            if (t2_v && t2_last) begin
                rq_busy <= 1'b1; rq_issue <= 1'b1; l_iss <= 7'd0; p_iss <= 2'd0;
                rq_obank <= t2_obank; rq_obase <= gch + t2_opix; oa <= gch + t2_opix;
            end

            // ---- A: latch the lane's four accumulators, address the bias/multiplier ROM
            ra_v <= 1'b0;
            if (rq_issue) begin
                pq_addr <= pqgrp + l_iss;
                ra_s0 <= sh[l_iss*4+0]; ra_s1 <= sh[l_iss*4+1]; ra_s2 <= sh[l_iss*4+2]; ra_s3 <= sh[l_iss*4+3];
                ra_p <= p_iss; ra_pool <= (c_MODE == M_POOL);
                ra_v <= 1'b1; ra_bank <= walk_bank_i; ra_addr <= oa;
                ra_last <= (l_iss == Lv - 1) && (p_iss == n_p_m1);
                if (l_iss == Lv - 1) begin
                    l_iss <= 7'd0; oa <= rq_obase;
                    if (p_iss == n_p_m1) rq_issue <= 1'b0; else p_iss <= p_iss + 1'b1;
                end else begin
                    l_iss <= l_iss + 1'b1; oa <= oa + c_BSZO;
                end
            end
            // ---- B: max (POOL) / select (QUAD, ONE); the ROM data arrives with this stage
            r0b_v <= ra_v; r0b_last <= ra_last; r0b_addr <= ra_addr; r0b_bank <= ra_bank; r0b_acc <= walk_val;
            // ---- R1: add bias
            r1_v <= r0b_v; r1_last <= r0b_last; r1_addr <= r0b_addr; r1_bank <= r0b_bank;
            r1_tot <= r0b_acc + $signed(pq_q[16 +: ACCW]);
            r1_m   <= pq_q[15:0];
            // ---- R2: requant multiply
            r2_v <= r1_v; r2_last <= r1_last; r2_addr <= r1_addr; r2_bank <= r1_bank; r2_tot <= r1_tot;
            r2_prod <= r1_tot * $signed({1'b0, r1_m});
            // ---- R3: round, shift, clamp
            r3_v <= r2_v; r3_last <= r2_last; r3_addr <= r2_addr; r3_bank <= r2_bank; r3_tot <= r2_tot;
            r3_y <= y_clamped;
            // ---- R4: write / logit
            if (r3_v) begin
                if (c_relu) begin
                    rq_we[r3_bank] <= 1'b1; rq_waddr <= r3_addr; rq_wdata <= r3_y;
                end else begin
                    logit <= {{(32-ACCW){r3_tot[ACCW-1]}}, r3_tot};
                end
                if (r3_last) rq_busy <= 1'b0;
            end

            // ---- main FSM
            case (st)
                S_IDLE, S_LOAD: begin
                    if (in_valid) begin
                        st <= S_LOAD;
                        if (in_kind == 2'd2) begin
                            if (side_cnt == N_SIDE - 1) begin
                                side_cnt <= 4'd0; got <= got | 3'b100;
                                if ((got | 3'b100) == 3'b111) begin got <= 3'b000; lyr <= 4'd0; st <= S_LCFG; end else st <= S_IDLE;
                            end else side_cnt <= side_cnt + 1'b1;
                        end else begin
                            if (ld_x == NIN - 1) begin
                                ld_x <= 7'd0;
                                if (ld_y[0]) ld_row <= ld_row + WB_NIN;
                                if (ld_y == NIN - 1) begin
                                    ld_y <= 7'd0; ld_row <= {AW{1'b0}};
                                    got <= got | (3'b001 << in_kind);
                                    if ((got | (3'b001 << in_kind)) == 3'b111) begin got <= 3'b000; lyr <= 4'd0; st <= S_LCFG; end else st <= S_IDLE;
                                end else ld_y <= ld_y + 1'b1;
                            end else ld_x <= ld_x + 1'b1;
                        end
                    end
                end
                S_LCFG: begin
                    c_K <= SEG_K(lyr, 0); c_SCIN <= SEG_CIN(lyr, 0); c_T <= L_T(lyr); c_MODE <= L_MODE(lyr); c_QH <= L_QH(lyr); c_QW <= L_QW(lyr);
                    c_WBIN <= SEG_WB(lyr, 0); c_BSZIN <= SEG_BSZ(lyr, 0); c_WBL <= L_WB_IN(lyr); c_WBO <= L_WB_O(lyr); c_BSZO <= L_BSZ_O(lyr);
                    c_relu <= L_relu(lyr); c_S <= L_S(lyr); c_G <= L_G(lyr); c_Cout <= L_Cout(lyr);
                    c_IMEM <= L_IMEM(lyr); c_OMEM <= L_OMEM(lyr); c_IBASE <= L_IBASE(lyr); c_OBASE <= L_OBASE(lyr); c_NSEG <= L_NSEG(lyr);
                    g <= 4'd0; gL <= 10'd0; gch <= L_OBASE(lyr);
                    wgrp <= L_WBASE(lyr); pqgrp <= L_PBASE(lyr);
                    rdsel <= L_IMEM(lyr);
                    st <= S_GRP;
                end
                S_GRP: begin
                    Lv <= (c_Cout - gL >= L) ? L : (c_Cout - gL);
                    qy <= 6'd0; qx <= 6'd0; base_pix <= {AW{1'b0}}; orow <= {AW{1'b0}}; cioff <= SEG_BASE(lyr, 0);
                    tcnt <= {TW{1'b0}}; kx <= 3'd0; ky <= 3'd0; wptr <= wgrp;
                    sg <= 2'd0; cch <= 10'd0; c_K <= SEG_K(lyr, 0); c_SCIN <= SEG_CIN(lyr, 0); c_WBIN <= SEG_WB(lyr, 0); c_BSZIN <= SEG_BSZ(lyr, 0);
                    st <= S_PIX;
                end
                S_PIX: begin
                    if (!stall) begin
                        i0_v <= 1'b1; i0_first <= (tcnt == 0); i0_last <= last_tap; i0_ky0 <= ky[0]; i0_kx0 <= kx[0];
                        i0_sr0 <= cioff + base_pix + rowmul(r0n, c_WBIN);
                        i0_sr1 <= cioff + base_pix + rowmul(r1n, c_WBIN);
                        i0_c0 <= c0n; i0_c1 <= c1n; i0_wa <= wptr; i0_opix <= opix_now; i0_obank <= obank_now;
                        if (!last_tap) begin
                            tcnt <= tcnt + 1'b1; wptr <= wptr + 1'b1;
                            if (kx != c_K - 1) kx <= kx + 1'b1;
                            else begin
                                kx <= 3'd0;
                                if (ky != c_K - 1) ky <= ky + 1'b1;
                                else begin
                                    ky <= 3'd0;
                                    if (cch == c_SCIN - 1) begin                      // segment finished (not the last one: !last_tap)
                                        sg <= sg + 1'b1; cch <= 10'd0;
                                        c_K <= SEG_K(lyr, sg + 1); c_SCIN <= SEG_CIN(lyr, sg + 1); c_WBIN <= SEG_WB(lyr, sg + 1);
                                        c_BSZIN <= SEG_BSZ(lyr, sg + 1); cioff <= SEG_BASE(lyr, sg + 1);
                                    end else begin cch <= cch + 1'b1; cioff <= cioff + c_BSZIN; end
                                end
                            end
                        end else begin
                            tcnt <= {TW{1'b0}}; wptr <= wgrp; kx <= 3'd0; ky <= 3'd0; cioff <= SEG_BASE(lyr, 0);
                            sg <= 2'd0; cch <= 10'd0; c_K <= SEG_K(lyr, 0); c_SCIN <= SEG_CIN(lyr, 0); c_WBIN <= SEG_WB(lyr, 0); c_BSZIN <= SEG_BSZ(lyr, 0);
                            if (!qx_last) begin
                                qx <= qx + 1'b1; base_pix <= base_pix + 1'b1;
                            end else begin
                                qx <= 6'd0;
                                if (!qy_last) begin
                                    qy <= qy + 1'b1;
                                    base_pix <= base_pix + c_WBL - (c_QW - 1);
                                    if (c_MODE != M_POOL || qy[0]) orow <= orow + c_WBO;
                                end
                            end
                            if (final_quad) st <= S_DRAIN;
                        end
                    end
                end
                S_DRAIN: begin
                    if (pipe_empty) begin
                        if (g == c_G - 1) begin
                            if (lyr == NL - 1) st <= S_DONE;
                            else begin lyr <= lyr + 1'b1; st <= S_LCFG; end
                        end else begin
                            g <= g + 1'b1; gL <= gL + L; gch <= gch + L * c_BSZO;
                            wgrp <= wgrp + c_T; pqgrp <= pqgrp + L;
                            st <= S_GRP;
                        end
                    end
                end
                S_DONE: begin
                    done   <= 1'b1;
                    detect <= (logit >= theta);
                    st     <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
