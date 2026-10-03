// cascade_ps_jtag_core.v -- pooled-domain Weibull prescreen + INT8 CNN cascade (cascade_ps_2clk) behind an Avalon-MM slave, for a host that
// supplies full 800x800 frames slowly (JTAG master from the PC) and reads the events back.  No HPS, no frame buffer.
//
// Unlike cascade_jtag_core there is NO clock gating: the stream domain now only holds the pooled-store writer, which tolerates gaps in
// the pixel stream; the Weibull prescreen runs afterwards on the stored frame.
//
// Avalon-MM slave, 32-bit words, word address:
//   0x0000 ID       R   0xCA5CADE3
//   0x0001 CTRL     W   bit0 = START_FRAME (clears results + counters, begins loading)   bit1 = SOFT_RESET
//   0x0002 CFG      RW  [1:0] pfa_sel (0: 3e-2  1: 1e-2  2: 1e-3  3: 1e-4)
//   0x0003 GTH      RW  prescreen gate on A = N*contrast/step  (config A, tau 0.6: 16065)
//   0x0004 THETA    RW  CNN decision threshold on the integer logit, signed
//   0x0005 STATUS   R   [0] ready (core idle, may START)  [1] loading  [2] frame_done (sticky)  [3] ev_overflow
//                       [4] pll_locked  [5] pixel words dropped (written outside a load)
//   0x0006 N_EVENTS R   candidate events of the frame   0x0007 N_ACCEPTED R  CNN-accepted events
//   0x0008 N_RES    R   result records captured         0x0009 PIX_ISSUED R   pixels delivered to the core this frame
//   0x000A POST_CYC R   clk cycles (50 MHz) from the last pixel to frame_done (prescreen + the whole CNN pass)
//   0x000B PS_CYC   R   clk_cnn cycles (100 MHz) spent in the prescreen pass
//   0x1000.. RESULTS R  (word addresses >= 0x1000; up to 2^EV_AW records)  record k = words 0x1000+2k (lo) and 0x1001+2k (hi):
//                       lo = {2'b0, accept, 9'b0, j[9:0], i[9:0]}   hi = signed logit
//   byte address >= 0x40000 PIXELS W  every write in this window pushes one word = 4 pixels, byte 0 (bits 7:0) first
//                       (the address inside the window is ignored, so the host may auto-increment it over a whole frame)
//   (word address = byte address >> 2; the top level passes avs_win = (byte address >= 0x40000))
`timescale 1ns/1ps
module cascade_ps_jtag_core #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter integer SLI = 25,
    parameter integer GUARD = 17,
    parameter QROM_HEX = "qrom.hex",
    parameter CNN_W_HEX  = "cnn_w.hex",
    parameter CNN_PQ_HEX = "cnn_pq.hex",
    parameter integer CNN_Q4 = 1,
    parameter integer EV_AW = 13,                // event / result RAM address bits (8192 events per frame)
    parameter [16:0] GTH_INIT = 17'd16065,
    parameter signed [31:0] THETA_INIT = -12312
) (
    input  wire        clk,                 // free-running 50 MHz
    input  wire        clk_cnn,             // 100 MHz PLL
    input  wire        rstn_in,             // board reset
    input  wire        pll_locked,
    // Avalon-MM slave (clk domain)
    input  wire [15:0] avs_address,
    input  wire        avs_win,             // 1: pixel window (any address >= 0x40000 bytes); avs_address is then ignored
    input  wire        avs_write,
    input  wire [31:0] avs_writedata,
    input  wire        avs_read,
    output reg  [31:0] avs_readdata,
    output reg         avs_readdatavalid,
    output wire        avs_waitrequest,
    // front-panel
    output wire        st_ready, st_loading, st_done, st_ovf,
    output wire [15:0] st_n_events, st_n_accepted
);
    localparam integer N_PIX = IMG_W * IMG_H;
    localparam [19:0] NPIX20 = N_PIX;

    // ------------------------------------------------------------------ soft reset
    reg [5:0] soft_cnt;
    wire core_rstn = rstn_in & pll_locked & (soft_cnt == 6'd0);

    // ------------------------------------------------------------------ configuration
    reg [1:0]  pfa_sel;
    reg [16:0] g_th;
    reg signed [31:0] theta;

    // ------------------------------------------------------------------ pixel feed + clock gate
    reg         loading;
    reg  [31:0] pw;
    reg  [2:0]  pcnt;                       // bytes left in `pw`
    reg  [19:0] bytes_left;
    reg  [19:0] pix_issued;
    reg  [7:0]  pixel_f;
    reg         valid_f;
    reg         clr_tog;
    reg         done_sticky;
    reg         drop_flag;
    reg         post_run;
    reg  [31:0] post_cyc;

    wire win_write = avs_write && avs_win;
    wire reg_write = avs_write && !avs_win;
    assign avs_waitrequest = win_write && loading && (pcnt != 3'd0);

    wire start_cmd = reg_write && (avs_address[7:0] == 8'h01) && avs_writedata[0];
    wire soft_cmd  = reg_write && (avs_address[7:0] == 8'h01) && avs_writedata[1];

    // ---- core
    wire        ready_for_frame, frame_done, ev_overflow;
    wire [15:0] n_events, n_accepted; wire [31:0] ps_cycles;
    wire        res_valid, res_accept; wire [9:0] res_j, res_i; wire signed [31:0] res_logit;
    cascade_ps_2clk #(.IMG_W(IMG_W), .IMG_H(IMG_H), .SLI(SLI), .GUARD(GUARD), .QROM_HEX(QROM_HEX),
                       .CNN_W_HEX(CNN_W_HEX), .CNN_PQ_HEX(CNN_PQ_HEX), .CNN_Q4(CNN_Q4), .EV_AW(EV_AW)) core (
        .clk(clk), .clk_cnn(clk_cnn), .rstn(core_rstn), .pixel_in_valid(valid_f), .pixel_in(pixel_f),
        .pfa_sel(pfa_sel), .g_th(g_th), .theta(theta),
        .ready_for_frame(ready_for_frame), .frame_done(frame_done), .n_events(n_events), .n_accepted(n_accepted),
        .ev_overflow(ev_overflow), .ps_cycles(ps_cycles), .res_valid(res_valid), .res_j(res_j), .res_i(res_i),
        .res_logit(res_logit), .res_accept(res_accept));

    always @(posedge clk or negedge rstn_in) begin
        if (!rstn_in) begin
            soft_cnt <= 6'd0; pfa_sel <= 2'd0; g_th <= GTH_INIT; theta <= THETA_INIT;
            loading <= 1'b0; pw <= 32'd0; pcnt <= 3'd0; bytes_left <= 20'd0; pix_issued <= 20'd0;
            pixel_f <= 8'd0; valid_f <= 1'b0; clr_tog <= 1'b0; done_sticky <= 1'b0; drop_flag <= 1'b0;
            post_run <= 1'b0; post_cyc <= 32'd0;
        end else begin
            if (soft_cnt != 6'd0) soft_cnt <= soft_cnt - 1'b1;
            if (soft_cmd) begin soft_cnt <= 6'd32; loading <= 1'b0; pcnt <= 3'd0; post_run <= 1'b0; end
            if (reg_write) case (avs_address[7:0])
                8'h02: pfa_sel <= avs_writedata[1:0];
                8'h03: g_th <= avs_writedata[16:0];
                8'h04: theta <= avs_writedata;
                default: ;
            endcase
            if (frame_done) begin done_sticky <= 1'b1; post_run <= 1'b0; end
            if (post_run) post_cyc <= post_cyc + 1'b1;

            // ---- pixel feed: output regs + clock enable (see the header)
            if (start_cmd) begin
                loading <= 1'b1; pcnt <= 3'd0; bytes_left <= NPIX20; pix_issued <= 20'd0;
                clr_tog <= ~clr_tog; done_sticky <= 1'b0; drop_flag <= 1'b0; post_run <= 1'b0; post_cyc <= 32'd0;
                valid_f <= 1'b0;
            end else if (loading && pcnt != 3'd0) begin
                pixel_f <= pw[7:0]; valid_f <= 1'b1;
                pw <= {8'd0, pw[31:8]}; pcnt <= pcnt - 1'b1;
                pix_issued <= pix_issued + 1'b1; bytes_left <= bytes_left - 1'b1;
                if (bytes_left == 20'd1) begin loading <= 1'b0; post_run <= 1'b1; post_cyc <= 32'd0; end
            end else begin
                valid_f <= 1'b0;
                if (win_write) begin
                    if (loading) begin pw <= avs_writedata; pcnt <= 3'd4; end
                    else drop_flag <= 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------ result RAM (written in clk_cnn, read in clk)
    reg [2:0] clr_s;
    reg [EV_AW:0] rwp;
    always @(posedge clk_cnn) clr_s <= {clr_s[1:0], clr_tog};
    wire clr_c = clr_s[2] ^ clr_s[1];
    wire [63:0] rwdata = {res_logit, 2'b00, res_accept, 9'd0, res_j, res_i};
    wire        rwe = res_valid && !rwp[EV_AW];
    always @(posedge clk_cnn or negedge core_rstn) begin
        if (!core_rstn) rwp <= {(EV_AW+1){1'b0}};
        else if (clr_c) rwp <= {(EV_AW+1){1'b0}};
        else if (rwe) rwp <= rwp + 1'b1;
    end
    reg [EV_AW:0] rwp_s1, rwp_s2;
    always @(posedge clk) begin rwp_s1 <= rwp; rwp_s2 <= rwp_s1; end

    wire [63:0] rrdata;
    reg  [EV_AW-1:0] raddr;
    wire [15:0] roff = avs_address - 16'h1000;
    event_ram_dc #(.WIDTH(64), .AW(EV_AW)) rram (
        .wclk(clk_cnn), .we(rwe), .waddr(rwp[EV_AW-1:0]), .wdata(rwdata),
        .rclk(clk), .raddr(raddr), .rdata(rrdata));

    // ------------------------------------------------------------------ Avalon reads (2-cycle latency)
    reg        rd1, rd2, rsel1, rsel2, rwin1, rwin2;
    reg [31:0] rreg1;
    wire [31:0] stat = {26'd0, drop_flag, pll_locked, ev_overflow, done_sticky, loading, (ready_for_frame && !loading)};
    always @(*) begin
        case (avs_address[7:0])
            8'h00: rreg1 = 32'hCA5CADE3;
            8'h02: rreg1 = {30'd0, pfa_sel};
            8'h03: rreg1 = {15'd0, g_th};
            8'h04: rreg1 = theta;
            8'h05: rreg1 = stat;
            8'h06: rreg1 = {16'd0, n_events};
            8'h07: rreg1 = {16'd0, n_accepted};
            8'h08: rreg1 = {{(31-EV_AW){1'b0}}, rwp_s2};
            8'h09: rreg1 = {12'd0, pix_issued};
            8'h0A: rreg1 = post_cyc;
            8'h0B: rreg1 = ps_cycles;
            default: rreg1 = 32'd0;
        endcase
    end
    reg [31:0] rreg_a, rreg_b;
    always @(posedge clk or negedge rstn_in) begin
        if (!rstn_in) begin
            rd1 <= 1'b0; rd2 <= 1'b0; rsel1 <= 1'b0; rsel2 <= 1'b0; rwin1 <= 1'b0; rwin2 <= 1'b0; rreg_a <= 32'd0; rreg_b <= 32'd0;
            avs_readdata <= 32'd0; avs_readdatavalid <= 1'b0; raddr <= {EV_AW{1'b0}};
        end else begin
            rd1 <= avs_read; rd2 <= rd1;
            rsel1 <= avs_address[0]; rsel2 <= rsel1;
            rwin1 <= (avs_address[15:12] != 4'h0); rwin2 <= rwin1;
            raddr <= roff[EV_AW:1];
            rreg_a <= rreg1; rreg_b <= rreg_a;
            avs_readdatavalid <= rd2;
            avs_readdata <= rwin2 ? (rsel2 ? rrdata[63:32] : rrdata[31:0]) : rreg_b;
        end
    end
    // read pipeline: request in cycle t -> sampled at edge t+1 (rd1/rsel1/rwin1/rreg_a/raddr) -> edge t+2 (rd2/..., RAM data)
    // -> edge t+3 (avs_readdata + readdatavalid).  Fixed 3-cycle latency; reads may be issued back to back.

    assign st_ready = ready_for_frame && !loading;
    assign st_loading = loading;
    assign st_done = done_sticky;
    assign st_ovf = ev_overflow;
    assign st_n_events = n_events;
    assign st_n_accepted = n_accepted;
endmodule
