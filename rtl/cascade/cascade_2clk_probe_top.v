// cascade_2clk_probe_top.v -- Quartus resource/timing probe for cascade_top_2clk with the real PLL (not a board top).
`timescale 1ns/1ps
module cascade_2clk_probe_top #(
    parameter integer IMG_W = 800, parameter integer IMG_H = 800,
    parameter LUT_ROOT = "lut", parameter QROM_HEX = "qrom.hex",
    parameter CNN_W_HEX = "cnn_w.hex", parameter CNN_PQ_HEX = "cnn_pq.hex"
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    input  wire [1:0]  pfa_sel,
    input  wire signed [18:0] tau_q14,
    input  wire signed [31:0] theta,
    output wire        ready_for_frame,
    output wire        frame_done,
    output wire [15:0] n_events,
    output wire [15:0] n_accepted,
    output wire        ev_overflow,
    output wire        res_valid,
    output wire        res_accept
);
    wire clk_cnn, locked;
    cnn_pll pll (.refclk(clk), .rst(1'b0), .outclk(clk_cnn), .locked(locked));
    wire [9:0] res_j, res_i; wire signed [31:0] res_logit;
    cascade_top_2clk #(.IMG_W(IMG_W), .IMG_H(IMG_H), .LUT_ROOT(LUT_ROOT), .QROM_HEX(QROM_HEX),
                       .CNN_W_HEX(CNN_W_HEX), .CNN_PQ_HEX(CNN_PQ_HEX), .CNN_Q4(1)) core (
        .clk(clk), .clk_cnn(clk_cnn), .rstn(rstn & locked), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .pfa_sel(pfa_sel), .tau_q14(tau_q14), .theta(theta), .ready_for_frame(ready_for_frame), .frame_done(frame_done),
        .n_events(n_events), .n_accepted(n_accepted), .ev_overflow(ev_overflow),
        .res_valid(res_valid), .res_j(res_j), .res_i(res_i), .res_logit(res_logit), .res_accept(res_accept));
endmodule
