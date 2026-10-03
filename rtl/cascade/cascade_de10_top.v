// cascade_de10_top.v -- DE10-Standard bring-up wrapper for the Weibull + CNN cascade (cascade_top.v).
//
// Self-contained (no HPS, no host): a 128x128 HRSID crop lives in on-chip ROM and is streamed through the
// cascade back-to-back, then the result is shown on the LEDs / 7-segment displays for ~2 s, and it repeats.
//
//   SW[1:0]  Pfa plane of the Weibull prescreen   (0: 1e-3 [the CNN training point], 1: 1e-4, 2: 1e-5, 3: 1e-6)
//   SW[3:2]  prescreen gate  g = x - c1 >= tau    (0: 0.60, 1: 0.70, 2: 0.75 [trained], 3: 0.80)
//   SW[5:4]  CNN operating point (val-selected threshold for ship retention 0: 80%, 1: 85%, 2: 90%, 3: 95%)
//   KEY[0]   board reset (active low)
//
// Clocking: the Weibull stream runs on CLOCK_50; the CNN stage runs on a 100 MHz PLL clock (cascade_top_2clk.v).
//   LEDR[0]  at least one candidate accepted by the CNN in the last frame
//   LEDR[1]  frame finished (pulse)          LEDR[2] streaming pixels        LEDR[3] heartbeat
//   LEDR[4]  CNN stage busy                  LEDR[5] event FIFO overflowed (sticky)
//   LEDR[9:6] accepted count (saturates at 15)
//   HEX5..3  candidate events this frame (hex)   HEX2..0  CNN-accepted events this frame (hex)
//
// The threshold table below is the INT8 model's own (cnn_weights_hw/<model>/manifest.json "thresholds_int").
`timescale 1ns/1ps
module cascade_de10_top #(
    parameter integer DISPLAY_CYCLES = 100_000_000,   // ~2 s at 50 MHz
    parameter integer CNN_Q4 = 1,                     // 1: quad-pixel CNN core (needs genq_* tables in CNN_W/PQ_HEX)
    parameter IMG_HEX   = "cascade_demo_image.hex",
    parameter CNN_W_HEX  = "F:/Projects/CFAR/rtl/cnn/genq_hw_deep_T/cnn_w.hex",
    parameter CNN_PQ_HEX = "F:/Projects/CFAR/rtl/cnn/genq_hw_deep_T/cnn_pq.hex",
    parameter signed [31:0] THETA0 = -9765,           // retention 80%
    parameter signed [31:0] THETA1 = -12312,          //           85%
    parameter signed [31:0] THETA2 = -15355,          //           90%
    parameter signed [31:0] THETA3 = -21068           //           95%
) (
    input  wire        CLOCK_50,
    input  wire [1:0]  KEY,
    input  wire [9:0]  SW,
    output wire [9:0]  LEDR,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5
);
    localparam integer IMG_W = 128, IMG_H = 128;
    localparam integer N_PIX = IMG_W * IMG_H;

    wire rstn = KEY[0];

    // synchronise the human inputs
    reg [9:0] sw_r;
    always @(posedge CLOCK_50 or negedge rstn)
        if (!rstn) sw_r <= 10'd0; else sw_r <= SW;

    reg signed [18:0] tau_q14;
    always @(*) begin
        case (sw_r[3:2])
            2'd0: tau_q14 = 19'sd9830;      // 0.60
            2'd1: tau_q14 = 19'sd11469;     // 0.70
            2'd2: tau_q14 = 19'sd12288;     // 0.75
            default: tau_q14 = 19'sd13107;  // 0.80
        endcase
    end
    reg signed [31:0] theta;
    always @(*) begin
        case (sw_r[5:4])
            2'd0: theta = THETA0;
            2'd1: theta = THETA1;
            2'd2: theta = THETA2;
            default: theta = THETA3;
        endcase
    end

    reg [7:0] img_rom [0:N_PIX-1];
    initial $readmemh(IMG_HEX, img_rom);

    reg        pixel_in_valid;
    reg [7:0]  pixel_in;
    wire       ready_for_frame, frame_done, ev_overflow, res_valid, res_accept;
    wire [9:0] res_j, res_i;
    wire signed [31:0] res_logit;
    wire [15:0] n_events, n_accepted;
    wire clk_cnn, pll_locked;
    cnn_pll pll (.refclk(CLOCK_50), .rst(1'b0), .outclk(clk_cnn), .locked(pll_locked));

    cascade_top_2clk #(.IMG_W(IMG_W), .IMG_H(IMG_H), .LUT_ROOT("F:/Projects/CFAR/lut"),
                       .QROM_HEX("F:/Projects/CFAR/rtl/cascade/qrom.hex"),
                       .CNN_W_HEX(CNN_W_HEX), .CNN_PQ_HEX(CNN_PQ_HEX), .CNN_Q4(CNN_Q4)) core (
        .clk(CLOCK_50), .clk_cnn(clk_cnn), .rstn(rstn & pll_locked), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(sw_r[1:0]), .tau_q14(tau_q14), .theta(theta),
        .ready_for_frame(ready_for_frame), .frame_done(frame_done), .n_events(n_events), .n_accepted(n_accepted),
        .ev_overflow(ev_overflow), .res_valid(res_valid), .res_j(res_j), .res_i(res_i),
        .res_logit(res_logit), .res_accept(res_accept));

    localparam [1:0] ST_WAIT = 2'd0, ST_STREAM = 2'd1, ST_RUN = 2'd2, ST_SHOW = 2'd3;
    reg [1:0]  st;
    reg [13:0] pix_addr;
    reg [31:0] show_cnt;
    reg [15:0] ev_shown, acc_shown;
    reg        frame_pulse;
    reg [25:0] heartbeat;

    always @(posedge CLOCK_50 or negedge rstn) begin
        if (!rstn) begin
            st <= ST_WAIT; pix_addr <= 14'd0; pixel_in_valid <= 1'b0; pixel_in <= 8'd0; show_cnt <= 32'd0;
            ev_shown <= 16'd0; acc_shown <= 16'd0; frame_pulse <= 1'b0;
        end else begin
            frame_pulse <= 1'b0;
            case (st)
                ST_WAIT: begin
                    pixel_in_valid <= 1'b0;
                    if (ready_for_frame) begin
                        pix_addr <= 14'd0; st <= ST_STREAM;
                    end
                end
                ST_STREAM: begin                      // back-to-back, no gaps (Weibull line buffer requirement)
                    pixel_in_valid <= 1'b1; pixel_in <= img_rom[pix_addr];
                    if (pix_addr == N_PIX - 1) st <= ST_RUN;      // last pixel IS driven this cycle; ST_RUN drops valid next cycle
                    else pix_addr <= pix_addr + 1'b1;
                end
                ST_RUN: begin
                    pixel_in_valid <= 1'b0;
                    if (frame_done) begin
                        ev_shown <= n_events; acc_shown <= n_accepted;
                        frame_pulse <= 1'b1; show_cnt <= 32'd0; st <= ST_SHOW;
                    end
                end
                ST_SHOW: begin
                    if (show_cnt < DISPLAY_CYCLES - 1) show_cnt <= show_cnt + 1'b1;
                    else st <= ST_WAIT;
                end
            endcase
        end
    end
    always @(posedge CLOCK_50 or negedge rstn)
        if (!rstn) heartbeat <= 26'd0; else heartbeat <= heartbeat + 1'b1;

    reg fifo_ovf_l;
    always @(posedge CLOCK_50 or negedge rstn)
        if (!rstn) fifo_ovf_l <= 1'b0; else if (ev_overflow) fifo_ovf_l <= 1'b1;

    wire [3:0] acc_sat = (acc_shown > 15) ? 4'd15 : acc_shown[3:0];
    assign LEDR[0] = (acc_shown != 0);
    assign LEDR[1] = frame_pulse;
    assign LEDR[2] = (st == ST_STREAM);
    assign LEDR[3] = heartbeat[25];
    assign LEDR[4] = (st == ST_RUN);
    assign LEDR[5] = fifo_ovf_l;
    assign LEDR[9:6] = acc_sat;

    function [6:0] seg;                    // active-low a..g
        input [3:0] v;
        case (v)
            4'h0: seg = 7'b1000000; 4'h1: seg = 7'b1111001; 4'h2: seg = 7'b0100100; 4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001; 4'h5: seg = 7'b0010010; 4'h6: seg = 7'b0000010; 4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000; 4'h9: seg = 7'b0010000; 4'hA: seg = 7'b0001000; 4'hB: seg = 7'b0000011;
            4'hC: seg = 7'b1000110; 4'hD: seg = 7'b0100001; 4'hE: seg = 7'b0000110; default: seg = 7'b0001110;
        endcase
    endfunction
    assign HEX0 = seg(acc_shown[3:0]);
    assign HEX1 = seg(acc_shown[7:4]);
    assign HEX2 = seg(acc_shown[11:8]);
    assign HEX3 = seg(ev_shown[3:0]);
    assign HEX4 = seg(ev_shown[7:4]);
    assign HEX5 = seg(ev_shown[11:8]);
endmodule
