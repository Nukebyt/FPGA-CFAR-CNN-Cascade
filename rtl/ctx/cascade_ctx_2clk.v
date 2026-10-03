// cascade_ctx_2clk.v -- two-pass cascade with the pooled-domain Weibull prescreen and the CONTEXT CNN (fine tower + context tower + side features).
//
//   pass 1 (clk, 50 MHz)       pixel stream -> QROM -> 2x2 pool -> pooled store;  img_stats accumulates the image-level statistics
//   pass 2 (clk_cnn, 100 MHz)  prescreen_top (events with A, S1, num>>12, P) -> side_unit (f0..f3) -> event RAM {j, i, f0..f3};
//                              img_codes (f4..f7 from the statistics) and ln_rom (f8 from the event count) run alongside
//   pass 3 (clk_cnn)           per event: 9 side bytes, fine patch (patch_fetch), context patch (ctx_fetch) -> cnn_core_ctx -> logit >= theta ?
// Clocking / handshakes as cascade_ps_2clk.v.
`timescale 1ns/1ps
module cascade_ctx_2clk #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter integer SLI = 25,
    parameter integer GUARD = 17,
    parameter QROM_HEX = "qrom.hex",
    parameter LN_HEX = "ln_rom.hex",
    parameter CNN_W_HEX  = "cnn_w.hex",
    parameter CNN_PQ_HEX = "cnn_pq.hex",
    parameter integer EV_AW = 12,
    parameter integer KC0 = 8381, parameter integer KC1 = 11060, parameter integer KC2 = 15733, parameter integer KC3 = 19546,
    parameter integer NUM_LO = 18371009, parameter integer NUM_HI = 1837100899
) (
    input  wire        clk,
    input  wire        clk_cnn,
    input  wire        rstn,
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    input  wire [1:0]  pfa_sel,
    input  wire [16:0] g_th,
    input  wire signed [31:0] theta,
    output wire        ready_for_frame,
    output reg         frame_done,
    output reg  [15:0] n_events,
    output reg  [15:0] n_accepted,
    output reg         ev_overflow,
    output reg  [31:0] ps_cycles,
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
    wire [27:0] st_sumq; wire [35:0] st_sumq2; wire [19:0] st_cb, st_cd;
    img_stats #(.QROM_HEX(QROM_HEX)) ist (.clk(clk), .rstn(core_rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .sumq(st_sumq), .sumq2(st_sumq2), .cnt_b(st_cb), .cnt_d(st_cd));

    localparam [2:0] SA_ACQ = 3'd0, SA_DLY = 3'd1, SA_WAIT = 3'd2, SA_CLR = 3'd3, SA_REL = 3'd4;
    reg [2:0] sst;
    reg pool_done_l, go;
    reg [3:0] dly;
    reg fin_c;
    reg fin_s1, fin_s2;
    wire fin_s = fin_s2;
    reg [15:0] n_acc_c, n_ev_cnt;
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
                SA_DLY:  begin dly <= dly + 1'b1; if (dly == 4'd7) begin go <= 1'b1; sst <= SA_WAIT; end end
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

    localparam [3:0] C_IDLE = 4'd0, C_PS = 4'd1, C_PSW = 4'd2, C_PSD = 4'd3, C_CHK = 4'd4, C_RD1 = 4'd5, C_RD2 = 4'd6, C_SIDE = 4'd7, C_FINE = 4'd8,
                     C_FINEW = 4'd9, C_CTX = 4'd10, C_CTXW = 4'd11, C_CNN = 4'd12, C_FIN = 4'd13, C_REL = 4'd14;
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

    // side features of every event, then the event RAM
    wire sv_o; wire [31:0] sv_codes;
    side_unit su (.clk(clk_cnn), .in_valid(pe_valid), .a(pe_cd), .s1(pe_pl[43:27]), .numsh(pe_pl[26:8]), .p(pe_pl[7:0]),
                  .out_valid(sv_o), .codes(sv_codes));
    reg [19:0] ji_d [0:8];
    integer ki;
    always @(posedge clk_cnn) begin
        ji_d[0] <= {pe_j, pe_i};
        for (ki = 1; ki < 9; ki = ki + 1) ji_d[ki] <= ji_d[ki-1];
    end
    reg [EV_AW:0] ev_wptr;
    wire ev_room = (ev_wptr < EV_DEPTH);
    wire [51:0] ev_rdata; reg [EV_AW-1:0] ev_raddr;
    event_ram_dc #(.WIDTH(52), .AW(EV_AW)) evram (
        .wclk(clk_cnn), .we(sv_o & ev_room), .waddr(ev_wptr[EV_AW-1:0]), .wdata({ji_d[8], sv_codes}),
        .rclk(clk_cnn), .raddr(ev_raddr), .rdata(ev_rdata));

    // image-level side codes
    wire [7:0] f4, f5, f6, f7, f8; wire ic_done;
    img_codes #(.NPIX(IMG_W * IMG_H)) icd (.clk(clk_cnn), .rstn(rstn_c), .start(ps_start), .sumq(st_sumq), .sumq2(st_sumq2), .cnt_b(st_cb), .cnt_d(st_cd),
                                          .f4(f4), .f5(f5), .f6(f6), .f7(f7), .done(ic_done));
    ln_rom #(.HEX(LN_HEX)) lnr (.clk(clk_cnn), .n((n_ev_cnt > 16'd8191) ? 13'd8191 : n_ev_cnt[12:0]), .code(f8));

    // pass 3: fetchers + CNN
    reg  pf_start, cf_start; reg [9:0] pf_j, pf_i; reg [31:0] ev_codes;
    wire pf_busy, pf_valid, cf_busy, cf_valid; wire [7:0] pf_data, cf_data; wire [ST_AW-1:0] pf_raddr, cf_raddr;
    wire sel_pre = (cst == C_PS) || (cst == C_PSW) || (cst == C_PSD);
    wire sel_cf  = (cst == C_CTX) || (cst == C_CTXW);
    wire [ST_AW-1:0] ps_raddr = sel_pre ? pre_raddr : sel_cf ? cf_raddr : pf_raddr;
    pooled_store_dc #(.DEPTH(ST_DEPTH), .AW(ST_AW)) store (
        .wclk(clk), .we(ps_we), .waddr(ps_waddr), .wdata(ps_wdata),
        .rclk(clk_cnn), .raddr(ps_raddr), .rdata(ps_rdata));
    patch_fetch #(.WP(WP), .HP(HP), .AW(ST_AW)) pf (
        .clk(clk_cnn), .rstn(rstn_c), .start(pf_start), .j(pf_j), .i(pf_i), .ps_raddr(pf_raddr), .ps_rdata(ps_rdata),
        .px_valid(pf_valid), .px_data(pf_data), .busy(pf_busy));
    ctx_fetch #(.WP(WP), .HP(HP), .AW(ST_AW)) cf (
        .clk(clk_cnn), .rstn(rstn_c), .start(cf_start), .j(pf_j), .i(pf_i), .ps_raddr(cf_raddr), .ps_rdata(ps_rdata),
        .px_valid(cf_valid), .px_data(cf_data), .busy(cf_busy));

    // side bytes: f0..f3 from the event, f4..f8 image-level
    reg [3:0] sb_cnt; reg sb_valid; reg [7:0] sb_data;
    wire [71:0] side_vec = {f8, f7, f6, f5, f4, ev_codes[31:24], ev_codes[23:16], ev_codes[15:8], ev_codes[7:0]};   // byte k = f_k
    wire in_valid = sb_valid | pf_valid | cf_valid;
    wire [7:0] in_data = sb_valid ? sb_data : pf_valid ? pf_data : cf_data;
    wire [1:0] in_kind = sb_valid ? 2'd2 : pf_valid ? 2'd0 : 2'd1;

    wire cnn_ready, cnn_done, cnn_detect; wire signed [31:0] cnn_logit;
    cnn_core_ctx #(.W_HEX(CNN_W_HEX), .PQ_HEX(CNN_PQ_HEX)) cnn (
        .clk(clk_cnn), .rstn(rstn_c), .in_valid(in_valid), .in_data(in_data), .in_kind(in_kind), .ready(cnn_ready),
        .done(cnn_done), .logit(cnn_logit), .theta(theta_c2), .detect(cnn_detect));

    reg [EV_AW:0] n_ev_c, rp;
    reg [4:0] wcnt;
    // cf_start is a one-clock pulse: ctx_fetch raises busy one clock later; guard against leaving C_CTXW before that
    reg cf_seen;
    wire pv_cf_pending = !cf_seen;
    always @(posedge clk_cnn) begin
        if (cst != C_CTXW) cf_seen <= 1'b0; else if (cf_busy) cf_seen <= 1'b1;
    end
    always @(posedge clk_cnn or negedge rstn_c) begin
        if (!rstn_c) begin
            cst <= C_IDLE; fin_c <= 1'b0; pf_start <= 1'b0; cf_start <= 1'b0; ps_start <= 1'b0; res_valid <= 1'b0; n_acc_c <= 16'd0; rp <= 0; n_ev_c <= 0;
            ev_raddr <= 0; res_j <= 10'd0; res_i <= 10'd0; res_logit <= 32'sd0; res_accept <= 1'b0; pf_j <= 10'd0; pf_i <= 10'd0;
            ev_wptr <= 0; n_ev_cnt <= 16'd0; ev_overflow <= 1'b0; ps_cyc_c <= 32'd0; sb_valid <= 1'b0; sb_cnt <= 4'd0; sb_data <= 8'd0; wcnt <= 5'd0; ev_codes <= 32'd0;
        end else begin
            pf_start <= 1'b0; cf_start <= 1'b0; ps_start <= 1'b0; res_valid <= 1'b0; sb_valid <= 1'b0;
            if (pe_valid) n_ev_cnt <= n_ev_cnt + 1'b1;
            if (sv_o) begin
                if (ev_room) ev_wptr <= ev_wptr + 1'b1; else ev_overflow <= 1'b1;
            end
            case (cst)
                C_IDLE: if (go_c2) begin ev_wptr <= 0; n_ev_cnt <= 16'd0; rp <= 0; n_acc_c <= 16'd0; ps_cyc_c <= 32'd0; ps_start <= 1'b1; cst <= C_PS; end
                C_PS:   begin ps_cyc_c <= ps_cyc_c + 1'b1; cst <= C_PSW; end
                C_PSW:  begin ps_cyc_c <= ps_cyc_c + 1'b1; if (pe_done) begin wcnt <= 5'd20; cst <= C_PSD; end end
                C_PSD:  begin                                      // let the side-unit pipeline drain, then latch the event count
                    if (wcnt == 0) begin n_ev_c <= ev_wptr; cst <= C_CHK; end else wcnt <= wcnt - 1'b1;
                end
                C_CHK: begin
                    if (rp == n_ev_c) cst <= C_FIN;
                    else if (cnn_ready) begin ev_raddr <= rp[EV_AW-1:0]; rp <= rp + 1'b1; cst <= C_RD1; end
                end
                C_RD1: cst <= C_RD2;
                C_RD2: begin pf_j <= ev_rdata[51:42]; pf_i <= ev_rdata[41:32]; ev_codes <= ev_rdata[31:0]; sb_cnt <= 4'd0; cst <= C_SIDE; end
                C_SIDE: begin                                     // nine side bytes, one per clock
                    sb_valid <= 1'b1; sb_data <= side_vec[8*sb_cnt +: 8];
                    if (sb_cnt == 4'd8) begin pf_start <= 1'b1; cst <= C_FINE; end else sb_cnt <= sb_cnt + 1'b1;
                end
                C_FINE: cst <= C_FINEW;
                C_FINEW: if (!pf_busy && !pf_valid) begin wcnt <= 5'd3; cst <= C_CTX; end
                C_CTX: begin if (wcnt == 0) begin cf_start <= 1'b1; cst <= C_CTXW; end else wcnt <= wcnt - 1'b1; end
                C_CTXW: if (cf_start == 1'b0 && !cf_busy && !cf_valid && !pv_cf_pending) cst <= C_CNN;
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
