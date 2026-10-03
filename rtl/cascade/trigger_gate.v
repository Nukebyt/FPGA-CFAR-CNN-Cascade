// trigger_gate.v -- turns the Weibull detection stream into CNN candidate events.
//
//   trigger  T(y,x) = D(y,x) & ~D(y,x-1) & ~D(y-1,x-1) & ~D(y-1,x) & ~D(y-1,x+1)
//            (the left end of a detected run with no detected neighbour above or to its left; the
//             streaming equivalent of "one candidate per connected detection blob", but needing only the
//             previous row of detect bits -- no connected-component labelling)
//   gate     g = x - c1 >= TAU    (both values already exist at the cell under test in the Weibull pipeline)
//   event    (j,i) = (y>>1, x>>1): position on the 2x2-pooled grid
//
// Input stream: one `det_valid` pulse per INTERIOR pixel (the Weibull core only outputs pixels whose
// 17x17 window fits), raster order, contiguous within a row. Coordinates are reconstructed by counting.
// NOTE: the Weibull core never emits the FIRST interior pixel (tb CMP_OFFSET=1, weibull_top_tb.v), so the
// first pulse of a frame is interior index 1, i.e. column TK+1 of row TK -> ix starts at 1.
//
// Fixed-point (Weibull core conventions): x_code is CENTRED Q1.14 (X0 already subtracted), c1_code is
// UNCENTRED Q?.13 (X0 added back), so  g_Q14 = x_code + X0_Q14 - 2*c1_code.
`timescale 1ns/1ps
module trigger_gate #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter integer TK    = 8,                  // (SLI-1)/2
    parameter integer X0_Q14 = 19866              // round(1.2125 * 2^14)
) (
    input  wire               clk,
    input  wire               rstn,
    input  wire               det_valid,
    input  wire               det,
    input  wire signed [15:0] x_code,
    input  wire signed [15:0] c1_code,
    input  wire signed [18:0] tau_q14,            // gate threshold, Q.14
    output reg                ev_valid,
    output reg  [9:0]         ev_j,
    output reg  [9:0]         ev_i,
    output reg                det_frame_done      // pulses with the last interior pixel
);
    localparam integer WI = IMG_W - 2*TK;
    localparam integer HI = IMG_H - 2*TK;

    reg [WI:0] dl;                  // dl[m] = det of the pulse m+1 pulses ago (delay line = previous row)
    reg [15:0] ix, iy;
    wire [15:0] xa = ix + TK;
    wire [15:0] ya = iy + TK;

    wire have_left   = (ix != 0);
    wire have_up     = (iy != 0);
    wire have_right  = (ix != WI - 1);
    wire left   = dl[0]  & have_left;
    wire aboveL = dl[WI] & have_up & have_left;
    wire above  = dl[WI-1] & have_up;
    wire aboveR = dl[WI-2] & have_up & have_right;
    wire trig   = det & ~left & ~aboveL & ~above & ~aboveR;

    wire signed [18:0] g_q14 = $signed(x_code) + X0_Q14 - ($signed(c1_code) <<< 1);
    wire pass = (g_q14 >= tau_q14);

    always @(posedge clk) begin
        if (!rstn) begin
            dl <= {(WI+1){1'b0}}; ix <= 16'd1; iy <= 16'd0;
            ev_valid <= 1'b0; ev_j <= 10'd0; ev_i <= 10'd0; det_frame_done <= 1'b0;
        end else begin
            ev_valid <= 1'b0;
            det_frame_done <= 1'b0;
            if (det_valid) begin
                dl <= {dl[WI-1:0], det};
                if (trig && pass) begin
                    ev_valid <= 1'b1;
                    ev_j <= ya[10:1];
                    ev_i <= xa[10:1];
                end
                if (ix == WI - 1) begin
                    ix <= 16'd0;
                    if (iy == HI - 1) begin
                        iy <= 16'd0; ix <= 16'd1;
                        det_frame_done <= 1'b1;
                        dl <= {(WI+1){1'b0}};
                    end else iy <= iy + 1'b1;
                end else ix <= ix + 1'b1;
            end
        end
    end
endmodule
