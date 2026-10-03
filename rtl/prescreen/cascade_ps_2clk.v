// cascade_ps_2clk.v -- two-pass cascade with the POOLED-DOMAIN Weibull prescreen (Paper 2, config A family).
//
//   pass 1  (clk, 50 MHz)      pixel stream -> QROM -> 2x2 pool -> pooled store                 (pool_store_writer, gap tolerant)
//   pass 2  (clk_cnn, 100 MHz) prescreen_top reads the stored pooled frame (mirrored borders), emits contrast-peak events
//   pass 3  (clk_cnn)          per event: patch_fetch (32x32 window of the store) -> cnn_core -> logit >= theta ?
//
// Compared with cascade_top_2clk.v the streaming Weibull core + trigger_gate (gap-intolerant, 17x17 window, 8-px unevaluated
// border) are replaced by prescreen_top, which runs after the frame is stored. Nothing in the stream domain needs a gap-free
// pixel stream any more, so a slow host (JTAG, HPS) can feed pixels directly.
//
// Clock-domain crossings: pooled_store_dc written in `clk`, read in `clk_cnn` (never overlapping: the CNN side starts only after the
// 4-phase go/fin handshake has crossed); `go` (clk -> clk_cnn) and `fin` (clk_cnn -> clk) are 2-FF synchronised levels;
// n_ev / n_acc are sampled by the clk side only after `fin`, while the CNN side is quiescent. pfa_sel, g_th and theta are static
// operating-point registers, double-registered into clk_cnn.
`timescale 1ns/1ps
module cascade_ps_2clk #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter integer SLI = 25,
    parameter integer GUARD = 17,
    parameter QROM_HEX = "qrom.hex",
    parameter CNN_W_HEX  = "cnn_w.hex",
    parameter CNN_PQ_HEX = "cnn_pq.hex",
    parameter integer EV_AW = 13,
    parameter integer CNN_Q4 = 1,
    parameter integer KC0 = 8381, parameter integer KC1 = 11060, parameter integer KC2 = 15733, parameter integer KC3 = 19546,
    parameter integer NUM_LO = 18371009, parameter integer NUM_HI = 1837100899
) (
    input  wire        clk,
    input  wire        clk_cnn,
    input  wire        rstn,                 // asynchronous, active low (also hold low until the PLL has locked)
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    input  wire [1:0]  pfa_sel,              // 0: 3e-2  1: 1e-2  2: 1e-3  3: 1e-4
    input  wire [16:0] g_th,                 // gate on A = N*(x-c1)/step  (config A: 16065 = contrast 0.6)
    input  wire signed [31:0] theta,
    output wire        ready_for_frame,
    output reg         frame_done,           // 1-cycle pulse in `clk`: the frame's CNN pass has finished
    output reg  [15:0] n_events,             // candidate events of the frame (valid from frame_done on)
    output reg  [15:0] n_accepted,
    output reg         ev_overflow,          // sticky: more events than the event RAM holds (extra events are dropped)
    output reg  [31:0] ps_cycles,            // clk_cnn cycles spent in the prescreen pass of the last frame
    // CNN-domain result stream (clk_cnn)
    output reg         res_valid,
    output reg  [9:0]  res_j,
    output reg  [9:0]  res_i,
    output reg  signed [31:0] res_logit,
    output reg         res_accept
);
    localparam integer WP = IMG_W / 2;
    localparam integer HP = IMG_H / 2;
    localparam integer ST_DEPTH = WP * HP;
    function integer clog2; input integer v; integer k; begin clog2 = 0; for (k = v - 1; k > 0; k = k >> 1) clog2 = clog2 + 1; end endfunction
    localparam integer ST_AW = clog2(ST_DEPTH);
    localparam integer EV_DEPTH = 1 << EV_AW;

    // ======================================================================= clk domain: pass 1
    reg [4:0] frm_cnt;
    reg       frame_rstn;
    wire      core_rstn = rstn & frame_rstn;

    wire ps_we; wire [ST_AW-1:0] ps_waddr; wire [7:0] ps_wdata; wire pool_done;
    pool_store_writer #(.IMG_W(IMG_W), .IMG_H(IMG_H), .QROM_HEX(QROM_HEX), .AW(ST_AW)) psw (
        .clk(clk), .rstn(core_rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .ps_we(ps_we), .ps_addr(ps_waddr), .ps_wdata(ps_wdata), .frame_in_done(pool_done));

    // stream-side frame controller
    localparam [2:0] SA_ACQ = 3'd0, SA_DLY = 3'd1, SA_WAIT = 3'd2, SA_CLR = 3'd3, SA_REL = 3'd4;
    reg [2:0] sst;
    reg pool_done_l, go;
    reg [3:0] dly;
    reg fin_c;                               // clk_cnn domain (declared here: used by the synchroniser below)
    reg fin_s1, fin_s2;
    wire fin_s = fin_s2;
    reg [15:0] n_acc_c, n_ev_cnt;            // clk_cnn domain; read here only while the CNN side is quiescent
    reg [31:0] ps_cyc_c;
    assign ready_for_frame = (sst == SA_ACQ) && frame_rstn && !pool_done_l;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin fin_s1 <= 1'b0; fin_s2 <= 1'b0; end
        else begin fin_s1 <= fin_c; fin_s2 <= fin_s1; end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sst <= SA_ACQ; frame_rstn <= 1'b1; frm_cnt <= 5'd0; pool_done_l <= 1'b0;
            go <= 1'b0; dly <= 4'd0; frame_done <= 1'b0; n_events <= 16'd0; n_accepted <= 16'd0; ps_cycles <= 32'd0;
        end else begin
            frame_done <= 1'b0;
            if (pool_done) pool_done_l <= 1'b1;
            case (sst)
                SA_ACQ:  if (pool_done_l) begin dly <= 4'd0; sst <= SA_DLY; end
                SA_DLY:  begin                                   // let the last store write settle before the other clock domain reads it
                    dly <= dly + 1'b1;
                    if (dly == 4'd7) begin go <= 1'b1; sst <= SA_WAIT; end
                end
                SA_WAIT: if (fin_s) begin
                    n_events <= n_ev_cnt; n_accepted <= n_acc_c; ps_cycles <= ps_cyc_c;
                    frame_done <= 1'b1; go <= 1'b0; frame_rstn <= 1'b0; frm_cnt <= 5'd0; sst <= SA_CLR;
                end
                SA_CLR:  begin
                    frm_cnt <= frm_cnt + 1'b1;
                    if (frm_cnt == 5'd16) begin frame_rstn <= 1'b1; pool_done_l <= 1'b0; sst <= SA_REL; end
                end
                SA_REL:  if (!fin_s) sst <= SA_ACQ;
                default: sst <= SA_ACQ;
            endcase
        end
    end

    // ===================================================================== clk_cnn domain
    reg rst_c1, rst_c2;
    always @(posedge clk_cnn or negedge rstn) begin
        if (!rstn) begin rst_c1 <= 1'b0; rst_c2 <= 1'b0; end
        else begin rst_c1 <= 1'b1; rst_c2 <= rst_c1; end
    end
    wire rstn_c = rst_c2;

    reg go_c1, go_c2;
    reg signed [31:0] theta_c1, theta_c2;
    reg [1:0]  pfa_c1, pfa_c2;
    reg [16:0] gth_c1, gth_c2;
    always @(posedge clk_cnn or negedge rstn_c) begin
        if (!rstn_c) begin go_c1 <= 1'b0; go_c2 <= 1'b0; theta_c1 <= 32'sd0; theta_c2 <= 32'sd0; pfa_c1 <= 2'd0; pfa_c2 <= 2'd0; gth_c1 <= 17'd0; gth_c2 <= 17'd0; end
        else begin go_c1 <= go; go_c2 <= go_c1; theta_c1 <= theta; theta_c2 <= theta_c1; pfa_c1 <= pfa_sel; pfa_c2 <= pfa_c1; gth_c1 <= g_th; gth_c2 <= gth_c1; end
    end

    localparam [3:0] C_IDLE = 4'd0, C_PS = 4'd1, C_PSW = 4'd2, C_CHK = 4'd3, C_RD1 = 4'd4, C_RD2 = 4'd5, C_FETCH = 4'd6, C_FWAIT = 4'd7,
                     C_CNN = 4'd8, C_FIN = 4'd9, C_REL = 4'd10;
    reg [3:0] cst;
    wire [7:0] ps_rdata;
    // pass 2: prescreen on the stored frame
    reg  ps_start;
    wire ps_busy, pe_valid, pe_done; wire [9:0] pe_j, pe_i; wire [16:0] pe_cd; wire [43:0] pe_pl;
    wire [ST_AW-1:0] pre_raddr;
    prescreen_top #(.WP(WP), .HP(HP), .SLI(SLI), .GUARD(GUARD), .AW(ST_AW), .KC0(KC0), .KC1(KC1), .KC2(KC2), .KC3(KC3),
                    .NUM_LO(NUM_LO), .NUM_HI(NUM_HI)) pre (
        .clk(clk_cnn), .rstn(rstn_c), .start(ps_start), .pfa_sel(pfa_c2), .g_th(gth_c2), .ps_raddr(pre_raddr), .ps_rdata(ps_rdata),
        .busy(ps_busy), .ev_valid(pe_valid), .ev_j(pe_j), .ev_i(pe_i), .ev_cd(pe_cd), .ev_pl(pe_pl), .done(pe_done));

    // event RAM (written in pass 2, read in pass 3; both in clk_cnn)
    reg [EV_AW:0]  ev_wptr;
    wire           ev_room = (ev_wptr < EV_DEPTH);
    wire [19:0] ev_rdata; reg [EV_AW-1:0] ev_raddr;
    event_ram_dc #(.WIDTH(20), .AW(EV_AW)) evram (
        .wclk(clk_cnn), .we(pe_valid & ev_room), .waddr(ev_wptr[EV_AW-1:0]), .wdata({pe_j, pe_i}),
        .rclk(clk_cnn), .raddr(ev_raddr), .rdata(ev_rdata));

    // pass 3: patch fetch + CNN
    reg  pf_start; reg [9:0] pf_j, pf_i;
    wire pf_busy, pf_valid; wire [7:0] pf_data; wire [ST_AW-1:0] pf_raddr;
    wire ps_sel_pre = (cst == C_PS) || (cst == C_PSW);           // the store read port belongs to the prescreen until its pipeline has drained (pe_done)
    wire [ST_AW-1:0] ps_raddr = ps_sel_pre ? pre_raddr : pf_raddr;
    pooled_store_dc #(.DEPTH(ST_DEPTH), .AW(ST_AW)) store (
        .wclk(clk), .we(ps_we), .waddr(ps_waddr), .wdata(ps_wdata),
        .rclk(clk_cnn), .raddr(ps_raddr), .rdata(ps_rdata));
    patch_fetch #(.WP(WP), .HP(HP), .AW(ST_AW)) pf (
        .clk(clk_cnn), .rstn(rstn_c), .start(pf_start), .j(pf_j), .i(pf_i), .ps_raddr(pf_raddr), .ps_rdata(ps_rdata),
        .px_valid(pf_valid), .px_data(pf_data), .busy(pf_busy));

    wire cnn_ready, cnn_done, cnn_detect; wire signed [31:0] cnn_logit;
    generate
        if (CNN_Q4) begin : g_q4
            cnn_core_q4 #(.W_HEX(CNN_W_HEX), .PQ_HEX(CNN_PQ_HEX)) cnn (
                .clk(clk_cnn), .rstn(rstn_c), .in_valid(pf_valid), .in_data(pf_data), .ready(cnn_ready),
                .done(cnn_done), .logit(cnn_logit), .theta(theta_c2), .detect(cnn_detect));
        end else begin : g_ser
            cnn_core #(.W_HEX(CNN_W_HEX), .PQ_HEX(CNN_PQ_HEX)) cnn (
                .clk(clk_cnn), .rstn(rstn_c), .in_valid(pf_valid), .in_data(pf_data), .ready(cnn_ready),
                .done(cnn_done), .logit(cnn_logit), .theta(theta_c2), .detect(cnn_detect));
        end
    endgenerate

    reg [EV_AW:0] n_ev_c, rp;
    always @(posedge clk_cnn or negedge rstn_c) begin
        if (!rstn_c) begin
            cst <= C_IDLE; fin_c <= 1'b0; pf_start <= 1'b0; ps_start <= 1'b0; res_valid <= 1'b0; n_acc_c <= 16'd0; rp <= 0; n_ev_c <= 0;
            ev_raddr <= 0; res_j <= 10'd0; res_i <= 10'd0; res_logit <= 32'sd0; res_accept <= 1'b0; pf_j <= 10'd0; pf_i <= 10'd0;
            ev_wptr <= 0; n_ev_cnt <= 16'd0; ev_overflow <= 1'b0; ps_cyc_c <= 32'd0;
        end else begin
            pf_start <= 1'b0; ps_start <= 1'b0; res_valid <= 1'b0;
            if (pe_valid) begin
                n_ev_cnt <= n_ev_cnt + 1'b1;
                if (ev_room) ev_wptr <= ev_wptr + 1'b1; else ev_overflow <= 1'b1;
            end
            case (cst)
                C_IDLE: if (go_c2) begin ev_wptr <= 0; n_ev_cnt <= 16'd0; rp <= 0; n_acc_c <= 16'd0; ps_cyc_c <= 32'd0; ps_start <= 1'b1; cst <= C_PS; end
                C_PS:   begin ps_cyc_c <= ps_cyc_c + 1'b1; cst <= C_PSW; end
                C_PSW:  begin
                    ps_cyc_c <= ps_cyc_c + 1'b1;
                    if (pe_done) begin n_ev_c <= ev_wptr; cst <= C_CHK; end
                end
                C_CHK: begin
                    if (rp == n_ev_c) cst <= C_FIN;
                    else if (cnn_ready) begin ev_raddr <= rp[EV_AW-1:0]; rp <= rp + 1'b1; cst <= C_RD1; end
                end
                C_RD1: cst <= C_RD2;                         // registered RAM read: address -> data takes two edges
                C_RD2: begin pf_j <= ev_rdata[19:10]; pf_i <= ev_rdata[9:0]; pf_start <= 1'b1; cst <= C_FETCH; end
                C_FETCH: cst <= C_FWAIT;
                C_FWAIT: if (!pf_busy) cst <= C_CNN;
                C_CNN: if (cnn_done) begin
                    res_valid <= 1'b1; res_j <= pf_j; res_i <= pf_i; res_logit <= cnn_logit; res_accept <= cnn_detect;
                    if (cnn_detect) n_acc_c <= n_acc_c + 1'b1;
                    cst <= C_CHK;
                end
                C_FIN: begin fin_c <= 1'b1; cst <= C_REL; end
                C_REL: if (!go_c2) begin fin_c <= 1'b0; cst <= C_IDLE; end
                default: cst <= C_IDLE;
            endcase
        end
    end
endmodule
