// cascade_top.v -- Weibull-CFAR prescreen + INT8 CNN discriminator, one chip.
//
//   pass 1 (stream):  pixel stream -> weibull_front_cascade -> trigger_gate -> event FIFO
//                                  \-> pool_store_writer    -> pooled frame store (1/4 size)
//   pass 2 (CNN):     for each event: patch_fetch (32x32 window, edge-replicated) -> cnn_core -> accept?
//
// The CNN stage starts when the frame has fully streamed (both the pooled store and the detection stream
// are complete). It is ~20-100x slower than the stream, which is why the frame is stored (pooled) and the
// events queued instead of trying to run the CNN at pixel rate.
//
// Control/status: after `ready_for_frame`, stream IMG_W*IMG_H pixels (gap-free, back to back; the Weibull
// line buffer requires that -- see KNOWN_ISSUE_GAP_INTOLERANCE.md). `res_valid` pulses once per processed
// event; `all_done` pulses when the last event has been classified. The Weibull core and counters are
// reset between frames (the line buffer has no frame boundary concept, BUG_LOG D20).
`timescale 1ns/1ps
module cascade_top #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter integer SLI = 17,
    parameter integer GUARD = 13,
    parameter LUT_ROOT = "lut",
    parameter QROM_HEX = "qrom.hex",
    parameter CNN_W_HEX  = "cnn_w.hex",
    parameter CNN_PQ_HEX = "cnn_pq.hex",
    parameter integer FIFO_AW = 10,
    parameter integer CNN_Q4 = 0               // 1: quad-pixel core (cnn_core_q4, needs the genq_* tables); 0: serial core
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    input  wire [1:0]  pfa_sel,
    input  wire signed [18:0] tau_q14,       // prescreen gate: (x - c1) >= tau, Q.14
    input  wire signed [31:0] theta,         // CNN decision threshold on the integer logit
    output wire        ready_for_frame,
    output reg         res_valid,
    output reg  [9:0]  res_j,
    output reg  [9:0]  res_i,
    output reg  signed [31:0] res_logit,
    output reg         res_accept,
    output reg         all_done,
    output reg  [15:0] n_events,
    output reg  [15:0] n_accepted,
    output wire        fifo_overflow
);
    localparam integer TK = (SLI - 1) / 2;
    localparam integer WP = IMG_W / 2;
    localparam integer HP = IMG_H / 2;
    localparam integer ST_DEPTH = WP * HP;
    function integer clog2; input integer v; integer k; begin clog2 = 0; for (k = v - 1; k > 0; k = k >> 1) clog2 = clog2 + 1; end endfunction
    localparam integer ST_AW = clog2(ST_DEPTH);

    // ---- frame-level reset (Weibull line buffer needs it between frames) ----------------------
    reg [4:0] frm_cnt;
    reg       frame_rstn;
    wire      core_rstn = rstn & frame_rstn;

    // ---- pass 1 -------------------------------------------------------------------------------
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

    wire [ST_AW-1:0] ps_raddr; wire [7:0] ps_rdata;
    pooled_store #(.DEPTH(ST_DEPTH), .AW(ST_AW)) store (
        .clk(clk), .we(ps_we), .waddr(ps_waddr), .wdata(ps_wdata), .raddr(ps_raddr), .rdata(ps_rdata));

    // ---- event queue ---------------------------------------------------------------------------
    reg  fifo_rd;
    wire [19:0] fifo_dout; wire fifo_dv, fifo_empty; wire [FIFO_AW:0] fifo_count;
    event_fifo #(.WIDTH(20), .AW(FIFO_AW)) efifo (
        .clk(clk), .rstn(rstn), .wr_en(ev_valid), .din({ev_j, ev_i}), .rd_en(fifo_rd),
        .dout(fifo_dout), .dout_valid(fifo_dv), .empty(fifo_empty), .count(fifo_count), .overflow(fifo_overflow));

    // ---- pass 2 --------------------------------------------------------------------------------
    reg  pf_start; reg [9:0] pf_j, pf_i;
    wire pf_busy, pf_valid; wire [7:0] pf_data;
    patch_fetch #(.WP(WP), .HP(HP), .AW(ST_AW)) pf (
        .clk(clk), .rstn(rstn), .start(pf_start), .j(pf_j), .i(pf_i), .ps_raddr(ps_raddr), .ps_rdata(ps_rdata),
        .px_valid(pf_valid), .px_data(pf_data), .busy(pf_busy));

    wire cnn_ready, cnn_done, cnn_detect; wire signed [31:0] cnn_logit;
    generate
        if (CNN_Q4) begin : g_q4
            cnn_core_q4 #(.W_HEX(CNN_W_HEX), .PQ_HEX(CNN_PQ_HEX)) cnn (
                .clk(clk), .rstn(rstn), .in_valid(pf_valid), .in_data(pf_data), .ready(cnn_ready),
                .done(cnn_done), .logit(cnn_logit), .theta(theta), .detect(cnn_detect));
        end else begin : g_ser
            cnn_core #(.W_HEX(CNN_W_HEX), .PQ_HEX(CNN_PQ_HEX)) cnn (
                .clk(clk), .rstn(rstn), .in_valid(pf_valid), .in_data(pf_data), .ready(cnn_ready),
                .done(cnn_done), .logit(cnn_logit), .theta(theta), .detect(cnn_detect));
        end
    endgenerate

    // ---- controller ----------------------------------------------------------------------------
    localparam [3:0] C_ACQ = 4'd0, C_POP = 4'd1, C_POPW = 4'd2, C_FETCH = 4'd3, C_FWAIT = 4'd4,
                     C_CNN = 4'd5, C_FIN = 4'd6, C_CLR = 4'd7;
    reg [3:0] cst;
    reg pool_done_l, det_done_l;
    assign ready_for_frame = (cst == C_ACQ) && frame_rstn && !pool_done_l;

    always @(posedge clk) begin
        if (!rstn) begin
            cst <= C_ACQ; frame_rstn <= 1'b1; frm_cnt <= 5'd0; pool_done_l <= 1'b0; det_done_l <= 1'b0;
            fifo_rd <= 1'b0; pf_start <= 1'b0; res_valid <= 1'b0; all_done <= 1'b0;
            n_events <= 16'd0; n_accepted <= 16'd0; res_j <= 10'd0; res_i <= 10'd0; res_logit <= 32'sd0; res_accept <= 1'b0;
        end else begin
            fifo_rd <= 1'b0; pf_start <= 1'b0; res_valid <= 1'b0; all_done <= 1'b0;
            if (pool_done) pool_done_l <= 1'b1;
            if (det_done)  det_done_l  <= 1'b1;
            if (ev_valid)  n_events <= n_events + 1'b1;
            case (cst)
                C_ACQ: begin
                    if (pool_done_l && det_done_l) cst <= C_POP;
                end
                C_POP: begin
                    if (!fifo_empty && cnn_ready) begin fifo_rd <= 1'b1; cst <= C_POPW; end
                    else if (fifo_empty) cst <= C_FIN;
                end
                C_POPW: begin
                    if (fifo_dv) begin pf_j <= fifo_dout[19:10]; pf_i <= fifo_dout[9:0]; pf_start <= 1'b1; cst <= C_FETCH; end
                end
                C_FETCH: cst <= C_FWAIT;                       // pf_start has been seen; busy rises next cycle
                C_FWAIT: begin
                    if (!pf_busy) cst <= C_CNN;
                end
                C_CNN: begin
                    if (cnn_done) begin
                        res_valid <= 1'b1; res_j <= pf_j; res_i <= pf_i; res_logit <= cnn_logit; res_accept <= cnn_detect;
                        if (cnn_detect) n_accepted <= n_accepted + 1'b1;
                        cst <= C_POP;
                    end
                end
                C_FIN: begin
                    all_done <= 1'b1; frame_rstn <= 1'b0; frm_cnt <= 5'd0; cst <= C_CLR;
                end
                C_CLR: begin
                    frm_cnt <= frm_cnt + 1'b1;
                    if (frm_cnt == 5'd16) begin
                        frame_rstn <= 1'b1; pool_done_l <= 1'b0; det_done_l <= 1'b0; cst <= C_ACQ;
                    end
                end
                default: cst <= C_ACQ;
            endcase
        end
    end
endmodule
