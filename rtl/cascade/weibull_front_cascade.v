// weibull_front_cascade.v -- the verified Weibull detector (same sub-modules/params as weibull_top_new.v,
// which is left untouched) with two extra outputs: the cell-under-test's log amplitude and the local
// log-mean c1, delayed so they are cycle-aligned with detect_valid/detect.
//   x_o  : centred Q1.14 (X0 subtracted)      c1_o : uncentred Q.13 (X0 added back)
`timescale 1ns/1ps
module weibull_front_cascade #(
    parameter integer SLI = 17,
    parameter integer GUARD = 13,
    parameter integer IMG_WIDTH  = 512,
    parameter integer IMG_HEIGHT = 512,
    parameter LUT_ROOT = "lut"
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    input  wire [1:0]  pfa_sel,
    output wire        detect_valid,
    output wire        detect,
    output wire signed [15:0] x_o,
    output wire signed [15:0] c1_o
);
    localparam integer BACKEND_LATENCY = 2;
    localparam integer DELTA_NATIVE_FRAC = 12;
    localparam integer DELTA_TARGET_FRAC = 10;
    localparam integer DELTA_SHIFT = DELTA_NATIVE_FRAC - DELTA_TARGET_FRAC;

    function automatic signed [15:0] rshift_round16;
        input signed [15:0] v; input integer sh;
        reg signed [31:0] rb;
        begin
            rb = (sh > 0) ? (32'sd1 <<< (sh-1)) : 32'sd0;
            rshift_round16 = (v >= 0) ? ((v + rb) >>> sh) : -((-v + rb) >>> sh);
        end
    endfunction

    wire        molc_valid;
    wire signed [15:0] c1_code;
    wire        [25:0] c2_code;
    wire signed [27:0] c3_code;
    wire signed [15:0] x_code;

    front_end3 #(.SLI(SLI), .GUARD(GUARD), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT),
        .LOG_AMP_HEX({LUT_ROOT, "/shared/log_amp_lut.hex"})) fe (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .molc_valid(molc_valid), .c1_code(c1_code), .c2_code(c2_code),
        .c3_code(c3_code), .x_code(x_code)
    );

    wire        be_valid;
    wire signed [15:0] delta_native;
    weibull_backend_new #(.LUT_ROOT(LUT_ROOT)) be (
        .clk(clk), .rstn(rstn), .molc_valid(molc_valid),
        .c2_code(c2_code), .pfa_sel(pfa_sel),
        .valid_out(be_valid), .delta_out(delta_native),
        .addr_sat_low(), .addr_sat_high()
    );
    wire signed [15:0] delta_code = rshift_round16(delta_native, DELTA_SHIFT);

    wire c1_dly_valid; wire signed [15:0] c1_dly;
    wire x_dly_valid;  wire signed [15:0] x_dly;
    pipe_delay #(.DATA_WIDTH(16), .DEPTH(BACKEND_LATENCY)) c1_delay (
        .clk(clk), .rstn(rstn), .in_valid(molc_valid), .in_data(c1_code),
        .out_valid(c1_dly_valid), .out_data(c1_dly));
    pipe_delay #(.DATA_WIDTH(16), .DEPTH(BACKEND_LATENCY)) x_delay (
        .clk(clk), .rstn(rstn), .in_valid(molc_valid), .in_data(x_code),
        .out_valid(x_dly_valid), .out_data(x_dly));

    wire signed [16:0] T_unused;
    threshold_compare3 tc (
        .clk(clk), .rstn(rstn), .in_valid(be_valid),
        .c1_code(c1_dly), .delta_code(delta_code), .x_code(x_dly),
        .detect_valid(detect_valid), .detect(detect), .T_log_code(T_unused));

    // threshold_compare3 registers its outputs: one more cycle to align x/c1 with detect_valid/detect
    pipe_delay #(.DATA_WIDTH(16), .DEPTH(1)) c1_align (
        .clk(clk), .rstn(rstn), .in_valid(c1_dly_valid), .in_data(c1_dly),
        .out_valid(), .out_data(c1_o));
    pipe_delay #(.DATA_WIDTH(16), .DEPTH(1)) x_align (
        .clk(clk), .rstn(rstn), .in_valid(x_dly_valid), .in_data(x_dly),
        .out_valid(), .out_data(x_o));
endmodule
