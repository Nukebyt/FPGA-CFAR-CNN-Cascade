// cascade_top_2clk.v -- cascade_top with the CNN stage in its own (faster) clock domain.
//
//   clk      (50 MHz, stream domain): Weibull core, trigger/gate, pooled-store writer, event writes
//   clk_cnn  (100 MHz, PLL):          patch fetch, CNN core, event reads, per-frame control
//
// Clock-domain crossings are deliberately few and simple:
//   * pooled_store_dc / event_ram_dc: written in `clk`, read in `clk_cnn`, never overlapping in time (the CNN pass
//     starts only after the frame has been written and the handshake below has crossed).
//   * `go`  (clk -> clk_cnn)  level, 2-FF synchronised: raised by the stream side once the frame is complete.
//   * `fin` (clk_cnn -> clk)  level, 2-FF synchronised: raised by the CNN side when every event is classified.
//     The classic 4-phase level handshake: go=1 .. fin=1 .. go=0 .. fin=0.
//   * Multi-bit values (event count, accepted count) are sampled only while the other side is quiescent
//     (event count: >= 8 `clk` cycles after the last event write, before the CNN reads it; accepted count: after
//     `fin` has been seen, while the CNN side is idle).
//   * theta is a static operating-point setting, double-registered into the CNN domain.
// Behaviour per frame is identical to cascade_top.v; only the clocking differs.
`timescale 1ns/1ps
module cascade_top_2clk #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter integer SLI = 17,
    parameter integer GUARD = 13,
    parameter LUT_ROOT = "lut",
    parameter QROM_HEX = "qrom.hex",
    parameter CNN_W_HEX  = "cnn_w.hex",
    parameter CNN_PQ_HEX = "cnn_pq.hex",
    parameter integer EV_AW = 10,
    parameter integer CNN_Q4 = 1
) (
    input  wire        clk,
    input  wire        clk_cnn,
    input  wire        rstn,                 // asynchronous, active low (also hold low until the PLL has locked)
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    input  wire [1:0]  pfa_sel,
    input  wire signed [18:0] tau_q14,
    input  wire signed [31:0] theta,
    output wire        ready_for_frame,
    output reg         frame_done,           // 1-cycle pulse in `clk`: the frame's CNN pass has finished
    output reg  [15:0] n_events,             // candidate events of the frame (valid from frame_done on)
    output reg  [15:0] n_accepted,           // CNN-accepted events of the frame
    output reg         ev_overflow,          // sticky: more events than the event RAM holds (extra events are dropped)
    // CNN-domain result stream (clk_cnn)
    output reg         res_valid,
    output reg  [9:0]  res_j,
    output reg  [9:0]  res_i,
    output reg  signed [31:0] res_logit,
    output reg         res_accept
);
    localparam integer TK = (SLI - 1) / 2;
    localparam integer WP = IMG_W / 2;
    localparam integer HP = IMG_H / 2;
    localparam integer ST_DEPTH = WP * HP;
    function integer clog2; input integer v; integer k; begin clog2 = 0; for (k = v - 1; k > 0; k = k >> 1) clog2 = clog2 + 1; end endfunction
    localparam integer ST_AW = clog2(ST_DEPTH);
    localparam integer EV_DEPTH = 1 << EV_AW;

    // ======================================================================= clk domain
    reg [4:0] frm_cnt;
    reg       frame_rstn;
    wire      core_rstn = rstn & frame_rstn;

    wire det_valid, det;
    wire signed [15:0] x_o, c1_o;
    weibull_front_cascade #(.SLI(SLI), .GUARD(GUARD), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H), .LUT_ROOT(LUT_ROOT)) wf (
        .clk(clk), .rstn(core_rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in), .pfa_sel(pfa_sel),
        .detect_valid(det_valid), .detect(det), .x_o(x_o), .c1_o(c1_o));

    wire ev_valid; wire [9:0] ev_j, ev_i; wire det_done;
    trigger_gate #(.IMG_W(IMG_W), .IMG_H(IMG_H), .TK(TK)) tg (
        .clk(clk), .rstn(core_rstn), .det_valid(det_valid), .det(det), .x_code(x_o), .c1_code(c1_o),
        .tau_q14(tau_q14), .ev_valid(ev_valid), .ev_j(ev_j), .ev_i(ev_i), .det_frame_done(det_done));

    wire ps_we; wire [ST_AW-1:0] ps_waddr; wire [7:0] ps_wdata; wire pool_done;
    pool_store_writer #(.IMG_W(IMG_W), .IMG_H(IMG_H), .QROM_HEX(QROM_HEX), .AW(ST_AW)) psw (
        .clk(clk), .rstn(core_rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .ps_we(ps_we), .ps_addr(ps_waddr), .ps_wdata(ps_wdata), .frame_in_done(pool_done));

    // event RAM write side
    reg [EV_AW:0]  ev_wptr;                 // events stored (<= EV_DEPTH)
    reg [15:0]     n_ev_cnt;                // events seen, including any dropped
    wire           ev_room = (ev_wptr < EV_DEPTH);
    always @(posedge clk) begin
        if (!core_rstn) begin ev_wptr <= 0; n_ev_cnt <= 16'd0; end
        else if (ev_valid) begin
            n_ev_cnt <= n_ev_cnt + 1'b1;
            if (ev_room) ev_wptr <= ev_wptr + 1'b1;
        end
    end
    always @(posedge clk or negedge rstn)
        if (!rstn) ev_overflow <= 1'b0; else if (ev_valid && !ev_room) ev_overflow <= 1'b1;

    wire [ST_AW-1:0] ps_raddr; wire [7:0] ps_rdata;
    pooled_store_dc #(.DEPTH(ST_DEPTH), .AW(ST_AW)) store (
        .wclk(clk), .we(ps_we), .waddr(ps_waddr), .wdata(ps_wdata),
        .rclk(clk_cnn), .raddr(ps_raddr), .rdata(ps_rdata));

    wire [19:0] ev_rdata; reg [EV_AW-1:0] ev_raddr;
    event_ram_dc #(.WIDTH(20), .AW(EV_AW)) evram (
        .wclk(clk), .we(ev_valid & ev_room), .waddr(ev_wptr[EV_AW-1:0]), .wdata({ev_j, ev_i}),
        .rclk(clk_cnn), .raddr(ev_raddr), .rdata(ev_rdata));

    // stream-side frame controller
    localparam [2:0] SA_ACQ = 3'd0, SA_DLY = 3'd1, SA_WAIT = 3'd2, SA_CLR = 3'd3, SA_REL = 3'd4;
    reg [2:0] sst;
    reg pool_done_l, det_done_l, go;
    reg [3:0] dly;
    reg fin_c;                               // clk_cnn domain (declared here: used by the synchroniser below)
    reg fin_s1, fin_s2;
    wire fin_s = fin_s2;
    reg [15:0] n_acc_c;                      // clk_cnn domain; read here only while the CNN side is quiescent
    assign ready_for_frame = (sst == SA_ACQ) && frame_rstn && !pool_done_l;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin fin_s1 <= 1'b0; fin_s2 <= 1'b0; end
        else begin fin_s1 <= fin_c; fin_s2 <= fin_s1; end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sst <= SA_ACQ; frame_rstn <= 1'b1; frm_cnt <= 5'd0; pool_done_l <= 1'b0; det_done_l <= 1'b0;
            go <= 1'b0; dly <= 4'd0; frame_done <= 1'b0; n_events <= 16'd0; n_accepted <= 16'd0;
        end else begin
            frame_done <= 1'b0;
            if (pool_done) pool_done_l <= 1'b1;
            if (det_done)  det_done_l  <= 1'b1;
            case (sst)
                SA_ACQ:  if (pool_done_l && det_done_l) begin dly <= 4'd0; sst <= SA_DLY; end
                SA_DLY:  begin                                   // let the last event write / counters settle before the CNN samples them
                    dly <= dly + 1'b1;
                    if (dly == 4'd7) begin go <= 1'b1; sst <= SA_WAIT; end
                end
                SA_WAIT: if (fin_s) begin
                    n_events <= n_ev_cnt; n_accepted <= n_acc_c;
                    frame_done <= 1'b1; go <= 1'b0; frame_rstn <= 1'b0; frm_cnt <= 5'd0; sst <= SA_CLR;
                end
                SA_CLR:  begin
                    frm_cnt <= frm_cnt + 1'b1;
                    if (frm_cnt == 5'd16) begin
                        frame_rstn <= 1'b1; pool_done_l <= 1'b0; det_done_l <= 1'b0; sst <= SA_REL;
                    end
                end
                SA_REL:  if (!fin_s) sst <= SA_ACQ;
                default: sst <= SA_ACQ;
            endcase
        end
    end

    // ===================================================================== clk_cnn domain
    // reset: asserted asynchronously, released synchronously
    reg rst_c1, rst_c2;
    always @(posedge clk_cnn or negedge rstn) begin
        if (!rstn) begin rst_c1 <= 1'b0; rst_c2 <= 1'b0; end
        else begin rst_c1 <= 1'b1; rst_c2 <= rst_c1; end
    end
    wire rstn_c = rst_c2;

    reg go_c1, go_c2;
    reg signed [31:0] theta_c1, theta_c2;
    always @(posedge clk_cnn or negedge rstn_c) begin
        if (!rstn_c) begin go_c1 <= 1'b0; go_c2 <= 1'b0; theta_c1 <= 32'sd0; theta_c2 <= 32'sd0; end
        else begin go_c1 <= go; go_c2 <= go_c1; theta_c1 <= theta; theta_c2 <= theta_c1; end
    end

    reg  pf_start; reg [9:0] pf_j, pf_i;
    wire pf_busy, pf_valid; wire [7:0] pf_data;
    patch_fetch #(.WP(WP), .HP(HP), .AW(ST_AW)) pf (
        .clk(clk_cnn), .rstn(rstn_c), .start(pf_start), .j(pf_j), .i(pf_i), .ps_raddr(ps_raddr), .ps_rdata(ps_rdata),
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

    localparam [3:0] C_IDLE = 4'd0, C_CHK = 4'd1, C_RD1 = 4'd2, C_RD2 = 4'd3, C_FETCH = 4'd4, C_FWAIT = 4'd5,
                     C_CNN = 4'd6, C_FIN = 4'd7, C_REL = 4'd8;
    reg [3:0] cst;
    reg [EV_AW:0] n_ev_c, rp;
    always @(posedge clk_cnn or negedge rstn_c) begin
        if (!rstn_c) begin
            cst <= C_IDLE; fin_c <= 1'b0; pf_start <= 1'b0; res_valid <= 1'b0; n_acc_c <= 16'd0; rp <= 0; n_ev_c <= 0;
            ev_raddr <= 0; res_j <= 10'd0; res_i <= 10'd0; res_logit <= 32'sd0; res_accept <= 1'b0; pf_j <= 10'd0; pf_i <= 10'd0;
        end else begin
            pf_start <= 1'b0; res_valid <= 1'b0;
            case (cst)
                C_IDLE: if (go_c2) begin n_ev_c <= ev_wptr; rp <= 0; n_acc_c <= 16'd0; cst <= C_CHK; end
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
