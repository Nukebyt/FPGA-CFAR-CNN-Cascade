// cnn_core.v -- INT8 CNN discriminator, layer-sequential MAC array.
//
// What it computes (bit-exact to cnn/quant_hw.py int_forward):
//   per layer:  acc[c] = sum_t w_int8[c][t] * a_uint8[t]              (int, ACCW bits)
//               tot    = acc + bias[c]
//               y      = clamp((tot * M[c] + (1 << (S-1))) >>> S, 0, 255)   (ReLU = lower clamp)
//               optional 2x2 max-pool over y (the four pooled pixels are computed back-to-back)
//   last layer: logit = tot (no requant); detect = (logit >= theta)
//
// Architecture
//   * L output-channel lanes, one int8 x uint8 MAC each (maps to a Cyclone V 9x9 DSP), fed by ONE
//     broadcast activation byte and one L*8-bit weight word per cycle.
//   * Two ping-pong activation RAMs (uint8). Layer n reads one and writes the other; layer 0 reads
//     RAM A, which is where the patch is loaded.
//   * Per output pixel the engine streams the T = Cin*K*K taps (weights come from a wide ROM, the
//     activation address from incremental adders -- no multipliers in the address path), then the
//     accumulators are copied to a shadow bank and a serial requant unit (one DSP multiplier) walks
//     the lanes while the next pixel is already accumulating. If T is too short for that overlap the
//     issue stage stalls on the last tap (never corrupts).
//   * Valid (un-padded) convolutions only, FC layers are conv with K=1 over the flattened [c][y][x]
//     activations. Weight/bias/shape tables come from gen_cnn_rtl.py (cnn_cfg.vh, cnn_w.hex, cnn_pq.hex).
//
// Interface: stream NIN*NIN bytes (row-major) on in_data/in_valid while `ready`; `done` pulses with
// `logit`/`detect` valid when the inference finishes.
`timescale 1ns/1ps
module cnn_core #(
    parameter W_HEX  = "cnn_w.hex",
    parameter PQ_HEX = "cnn_pq.hex",
    parameter integer ACCW = 26
) (
    input  wire               clk,
    input  wire               rstn,
    input  wire               in_valid,
    input  wire [7:0]         in_data,
    output wire               ready,
    output reg                done,
    output reg  signed [31:0] logit,
    input  wire signed [31:0] theta,
    output reg                detect
);
`include "cnn_cfg.vh"
    localparam integer L = LANES_GEN;

    // ------------------------------------------------------------------ state
    localparam [3:0] S_IDLE = 4'd0, S_LOAD = 4'd1, S_LCFG = 4'd2, S_GRP = 4'd3,
                     S_PIX = 4'd4, S_DRAIN = 4'd5, S_DONE = 4'd6;
    reg [3:0] st;
    assign ready = (st == S_IDLE);

    reg [AW-1:0]  ld_cnt;
    reg [3:0]     lyr;
    reg [3:0]     g;
    reg [9:0]     gL;           // g*L
    reg [AW-1:0]  gch;          // output offset of channel group g
    reg [WAW-1:0] wgrp;         // weight word index of (layer, group, tap 0)
    reg [PQAW-1:0] pqgrp;
    reg [6:0]     Lv;           // valid lanes in this group

    // layer configuration registers
    reg [3:0]     c_K;
    reg [TW-1:0]  c_T;
    reg [AW-1:0]  c_INC_KY, c_INC_CI, c_SP2, c_STEP_X, c_STEP_ROW, c_OUTHW;
    reg [5:0]     c_GW, c_GH;
    reg           c_pool, c_relu;
    reg [5:0]     c_S;
    reg [3:0]     c_G;
    reg [9:0]     c_Cout;

    // issue-engine state
    reg [5:0]     px, py;
    reg [1:0]     sp;
    reg [AW-1:0]  org, Po, t_addr, opix;
    reg [TW-1:0]  tcnt;
    reg [2:0]     kx, ky;
    reg [WAW-1:0] wptr;

    // ------------------------------------------------------------ pipelines
    reg [AW-1:0]  a_addr;
    reg [WAW-1:0] wa_addr;
    reg           iss_v, iss_first, iss_last;
    reg [1:0]     iss_sp;
    reg [AW-1:0]  iss_opix;
    reg           t1_v, t1_first, t1_last;
    reg [1:0]     t1_sp;
    reg [AW-1:0]  t1_opix;
    reg           t2_v, t2_first, t2_last;
    reg [1:0]     t2_sp;
    reg [AW-1:0]  t2_opix;

    // activation RAMs
    reg [7:0] memA [0:ABUF_DEPTH-1];
    reg [7:0] memB [0:ABUF_DEPTH-1];
    reg [7:0] qA, qB;
    reg       rdsel;                 // 1: layer reads RAM B
    wire [7:0] act_q = rdsel ? qB : qA;

    // weight / bias+multiplier ROMs
    reg [L*8-1:0] wmem [0:WDEPTH-1];
    reg [47:0]    pqmem [0:PQDEPTH-1];
    initial begin
        $readmemh(W_HEX, wmem);
        $readmemh(PQ_HEX, pqmem);
    end
    reg [L*8-1:0] w_q;
    reg [47:0]    pq_q;
    reg [PQAW-1:0] pq_addr;

    // write port (shared: patch loader / requant writer)
    wire ld_we = in_valid && (st == S_IDLE || st == S_LOAD);
    wire [AW-1:0] ld_addr = (st == S_IDLE) ? {AW{1'b0}} : ld_cnt;
    reg           rq_we;
    reg [AW-1:0]  rq_waddr;
    reg [7:0]     rq_wdata;
    wire          wA_en = ld_we | (rq_we & lyr[0]);
    wire          wB_en = rq_we & ~lyr[0];
    wire [AW-1:0] w_addr_o = ld_we ? ld_addr : rq_waddr;
    wire [7:0]    w_data_o = ld_we ? in_data : rq_wdata;

    always @(posedge clk) begin
        if (wA_en) memA[w_addr_o] <= w_data_o;
        qA <= memA[a_addr];
    end
    always @(posedge clk) begin
        if (wB_en) memB[w_addr_o] <= w_data_o;
        qB <= memB[a_addr];
    end
    always @(posedge clk) begin
        w_q  <= wmem[wa_addr];
        pq_q <= pqmem[pq_addr];
    end

    // ----------------------------------------------------- MAC lanes (int8 x uint8)
    reg signed [17:0]       prod [0:L-1];
    reg signed [ACCW-1:0]   acc  [0:L-1];
    reg signed [ACCW-1:0]   sh   [0:L-1];     // shadow bank for the serial requant unit
    genvar gl;
    generate
        for (gl = 0; gl < L; gl = gl + 1) begin : lane
            wire signed [8:0] a9 = {1'b0, act_q};
            wire signed [7:0] w8 = w_q[8*gl +: 8];
            always @(posedge clk) begin
                if (t1_v) prod[gl] <= a9 * w8;
            end
            wire signed [ACCW-1:0] nxt = t2_first ? $signed(prod[gl])
                                                  : (acc[gl] + $signed(prod[gl]));
            always @(posedge clk) begin
                if (t2_v) begin
                    acc[gl] <= nxt;
                    if (t2_last) sh[gl] <= nxt;
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------- requant unit
    reg           rq_busy, rq_issue;
    reg [6:0]     l_iss;
    reg [PQAW-1:0] pq_ptr;
    reg [1:0]     rq_sp;
    reg [AW-1:0]  rq_oaddr;
    // R0
    reg                    r0_v, r0_last;
    reg [6:0]              r0_lane;
    reg signed [ACCW-1:0]  r0_acc;
    // R0b: waits one cycle for the synchronous bias/multiplier ROM read
    reg                    r0b_v, r0b_last;
    reg [6:0]              r0b_lane;
    reg signed [ACCW-1:0]  r0b_acc;
    // R1
    reg                    r1_v, r1_last;
    reg [6:0]              r1_lane;
    reg signed [ACCW-1:0]  r1_tot;
    reg [15:0]             r1_m;
    // R2
    reg                    r2_v, r2_last;
    reg [6:0]              r2_lane;
    reg signed [ACCW-1:0]  r2_tot;
    reg signed [ACCW+16:0] r2_prod;
    // R3
    reg                    r3_v, r3_last;
    reg [6:0]              r3_lane;
    reg signed [ACCW-1:0]  r3_tot;
    reg [7:0]              r3_y;
    reg [7:0] pool_acc [0:L-1];

    wire signed [ACCW+16:0] rnd = (c_S == 0) ? {(ACCW+17){1'b0}} : ({{(ACCW+16){1'b0}}, 1'b1} <<< (c_S - 1));
    wire signed [ACCW+16:0] r2_rnd = r2_prod + rnd;
    wire signed [ACCW+16:0] r2_shr = r2_rnd >>> c_S;
    wire [7:0] y_clamped = (r2_shr < 0) ? 8'd0 : (r2_shr > 255) ? 8'd255 : r2_shr[7:0];

    // ------------------------------------------------------------ issue control
    wire px_last = (px == c_GW - 1);
    wire py_last = (py == c_GH - 1);
    wire final_pix = (c_pool ? (sp == 2'd3) : 1'b1) && px_last && py_last;
    wire last_tap  = (tcnt == c_T - 1);
    wire last_inflight = (iss_v & iss_last) | (t1_v & t1_last) | (t2_v & t2_last);
    wire stall = last_tap & (rq_busy | last_inflight);
    wire pipe_empty = ~iss_v & ~t1_v & ~t2_v & ~rq_busy;

    reg [AW-1:0] norg;
    reg [7:0]    newm;

    always @(posedge clk) begin
        if (!rstn) begin
            st <= S_IDLE; done <= 1'b0; detect <= 1'b0; logit <= 32'sd0;
            iss_v <= 1'b0; t1_v <= 1'b0; t2_v <= 1'b0;
            rq_busy <= 1'b0; rq_issue <= 1'b0; r0_v <= 1'b0; r0b_v <= 1'b0; r1_v <= 1'b0; r2_v <= 1'b0; r3_v <= 1'b0;
            rq_we <= 1'b0; ld_cnt <= {AW{1'b0}}; lyr <= 4'd0;
        end else begin
            done  <= 1'b0;
            rq_we <= 1'b0;
            // pipeline flags
            t1_v <= iss_v; t1_first <= iss_first; t1_last <= iss_last; t1_sp <= iss_sp; t1_opix <= iss_opix;
            t2_v <= t1_v;  t2_first <= t1_first;  t2_last <= t1_last;  t2_sp <= t1_sp;  t2_opix <= t1_opix;
            iss_v <= 1'b0;

            // ---- capture: last tap of a pixel is being accumulated -> start the requant walk
            if (t2_v && t2_last) begin
                rq_busy  <= 1'b1;
                rq_issue <= 1'b1;
                l_iss    <= 7'd0;
                rq_sp    <= t2_sp;
                pq_ptr   <= pqgrp;
                rq_oaddr <= gch + t2_opix;
            end

            // ---- R0: pick lane accumulator, address the bias/multiplier ROM
            r0_v <= 1'b0;
            if (rq_issue) begin
                pq_addr <= pq_ptr;
                pq_ptr  <= pq_ptr + 1'b1;
                r0_acc  <= sh[l_iss];
                r0_v    <= 1'b1;
                r0_lane <= l_iss;
                r0_last <= (l_iss == Lv - 1);
                l_iss   <= l_iss + 1'b1;
                if (l_iss == Lv - 1) rq_issue <= 1'b0;
            end
            // ---- R0b: ROM data arrives
            r0b_v <= r0_v; r0b_last <= r0_last; r0b_lane <= r0_lane; r0b_acc <= r0_acc;
            // ---- R1: add bias
            r1_v <= r0b_v; r1_last <= r0b_last; r1_lane <= r0b_lane;
            r1_tot <= r0b_acc + $signed(pq_q[16 +: ACCW]);
            r1_m   <= pq_q[15:0];
            // ---- R2: multiply by the requant multiplier
            r2_v <= r1_v; r2_last <= r1_last; r2_lane <= r1_lane; r2_tot <= r1_tot;
            r2_prod <= r1_tot * $signed({1'b0, r1_m});
            // ---- R3: round, shift, clamp
            r3_v <= r2_v; r3_last <= r2_last; r3_lane <= r2_lane; r3_tot <= r2_tot;
            r3_y <= y_clamped;
            // ---- R4: pool / write / logit
            if (r3_v) begin
                if (c_relu) begin
                    if (c_pool) begin
                        if (rq_sp == 2'd0) begin
                            pool_acc[r3_lane] <= r3_y;
                        end else begin
                            newm = (r3_y > pool_acc[r3_lane]) ? r3_y : pool_acc[r3_lane];
                            if (rq_sp == 2'd3) begin
                                rq_we <= 1'b1; rq_waddr <= rq_oaddr; rq_wdata <= newm;
                            end else begin
                                pool_acc[r3_lane] <= newm;
                            end
                        end
                    end else begin
                        rq_we <= 1'b1; rq_waddr <= rq_oaddr; rq_wdata <= r3_y;
                    end
                    rq_oaddr <= rq_oaddr + c_OUTHW;
                end else begin
                    logit <= {{(32-ACCW){r3_tot[ACCW-1]}}, r3_tot};
                end
                if (r3_last) rq_busy <= 1'b0;
            end

            // ---- main FSM
            case (st)
                S_IDLE: begin
                    if (in_valid) begin
                        ld_cnt <= {{(AW-1){1'b0}}, 1'b1};
                        st <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    if (in_valid) begin
                        ld_cnt <= ld_cnt + 1'b1;
                        if (ld_cnt == NIN*NIN - 1) begin
                            lyr <= 4'd0;
                            st  <= S_LCFG;
                        end
                    end
                end
                S_LCFG: begin
                    c_K <= L_K(lyr); c_T <= L_T(lyr); c_INC_KY <= L_INC_KY(lyr); c_INC_CI <= L_INC_CI(lyr);
                    c_SP2 <= L_SP2(lyr); c_STEP_X <= L_STEP_X(lyr); c_STEP_ROW <= L_STEP_ROW(lyr);
                    c_OUTHW <= L_OUT_HW(lyr); c_GW <= L_GW(lyr); c_GH <= L_GH(lyr);
                    c_pool <= L_pool(lyr); c_relu <= L_relu(lyr); c_S <= L_S(lyr);
                    c_G <= L_G(lyr); c_Cout <= L_Cout(lyr);
                    g <= 4'd0; gL <= 10'd0; gch <= {AW{1'b0}};
                    wgrp <= L_WBASE(lyr); pqgrp <= L_PBASE(lyr);
                    rdsel <= lyr[0];
                    st <= S_GRP;
                end
                S_GRP: begin
                    Lv <= (c_Cout - gL >= L) ? L : (c_Cout - gL);
                    px <= 6'd0; py <= 6'd0; sp <= 2'd0; opix <= {AW{1'b0}};
                    org <= {AW{1'b0}}; Po <= {AW{1'b0}}; t_addr <= {AW{1'b0}};
                    tcnt <= {TW{1'b0}}; kx <= 3'd0; ky <= 3'd0; wptr <= wgrp;
                    st <= S_PIX;
                end
                S_PIX: begin
                    if (!stall) begin
                        a_addr <= t_addr; wa_addr <= wptr;
                        iss_v <= 1'b1; iss_first <= (tcnt == 0); iss_last <= last_tap;
                        iss_sp <= sp; iss_opix <= opix;
                        if (!last_tap) begin
                            tcnt <= tcnt + 1'b1; wptr <= wptr + 1'b1;
                            if (kx != c_K - 1) begin
                                kx <= kx + 1'b1; t_addr <= t_addr + 1'b1;
                            end else begin
                                kx <= 3'd0;
                                if (ky != c_K - 1) begin
                                    ky <= ky + 1'b1; t_addr <= t_addr + c_INC_KY;
                                end else begin
                                    ky <= 3'd0; t_addr <= t_addr + c_INC_CI;
                                end
                            end
                        end else begin
                            tcnt <= {TW{1'b0}}; wptr <= wgrp; kx <= 3'd0; ky <= 3'd0;
                            if (c_pool) begin
                                if (sp != 2'd3) begin
                                    sp <= sp + 1'b1;
                                    norg = (sp == 2'd1) ? (org + c_SP2) : (org + 1'b1);
                                    org <= norg; t_addr <= norg;
                                end else begin
                                    sp <= 2'd0; opix <= opix + 1'b1;
                                    norg = px_last ? (Po + c_STEP_ROW) : (Po + c_STEP_X);
                                    Po <= norg; org <= norg; t_addr <= norg;
                                    if (px_last) begin px <= 6'd0; py <= py + 1'b1; end
                                    else px <= px + 1'b1;
                                end
                            end else begin
                                opix <= opix + 1'b1;
                                norg = px_last ? (org + c_STEP_ROW) : (org + c_STEP_X);
                                org <= norg; t_addr <= norg;
                                if (px_last) begin px <= 6'd0; py <= py + 1'b1; end
                                else px <= px + 1'b1;
                            end
                            if (final_pix) st <= S_DRAIN;
                        end
                    end
                end
                S_DRAIN: begin
                    if (pipe_empty && !t2_v) begin
                        if (g == c_G - 1) begin
                            if (lyr == NL - 1) begin
                                st <= S_DONE;
                            end else begin
                                lyr <= lyr + 1'b1;
                                st <= S_LCFG;
                            end
                        end else begin
                            g <= g + 1'b1; gL <= gL + L; gch <= gch + L * c_OUTHW;
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
