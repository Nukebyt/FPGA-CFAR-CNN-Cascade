// front_end3.v
// ---------------------------------------------------------------------------
// The shared 3-moment CFAR front end (FINDINGS F1: one shared front end, a
// swappable back-end ROM per detector). Pipeline:
//
//   pixel_in (8-bit) -> log_amp_rom -> power_expand -> {xc,xc2,xc3}
//     -> 3x line_buffer (one per plane, driven in lockstep)
//     -> 3x box_sum_inc (window-minus-guard sum per plane, adds only)
//     -> molc_estimator3 -> c1_code, c2_code, c3_code, molc_valid
//
// The cell-under-test's centred log-amplitude (x_code, for the final
// threshold compare in each detector's back end) is read directly out of the
// x-plane window snapshot's CENTRE element -- not re-derived through a
// separately hand-computed delay depth. This sidesteps exactly the class of
// bug line_buffer.v's own header describes at length (a hand-derived warmup
// formula that looked right and was not): the centre pixel is ALREADY present,
// on the SAME cycle as window_valid, as one static bit-slice of the x-plane
// window bus. It only needs delaying by box_sum_inc's + molc_estimator3's OWN
// (exactly known, fixed) latency to land on the same cycle as molc_valid.
// ---------------------------------------------------------------------------

module front_end3 #(
    parameter integer SLI = 17,
    parameter integer GUARD = 13,
    parameter integer IMG_WIDTH  = 512,
    parameter integer IMG_HEIGHT = 512,
    parameter DW  = 16,   // xc,  Q1.14 signed
    parameter X2W = 26,   // xc2, Q2.24 unsigned
    parameter X3W = 27,   // xc3, Q2.24 signed
    parameter integer S1W = 23, parameter integer S2W = 33, parameter integer S3W = 34,
    parameter integer C1W = 16, parameter integer C2W = 26, parameter integer C3W = 28,
    parameter LOG_AMP_HEX = "lut/shared/log_amp_lut.hex"
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,

    output wire                   molc_valid,
    output wire signed [C1W-1:0]  c1_code,
    output wire        [C2W-1:0]  c2_code,
    output wire signed [C3W-1:0]  c3_code,
    output wire signed [DW-1:0]   x_code      // cell-under-test, aligned with molc_valid
);

    localparam integer TK_SLI = (SLI-1)/2;

    // ---- Stage: log-amplitude lookup (1 cycle) -----------------------------
    wire signed [DW-1:0] xc_lut;
    log_amp_rom #(.HEX_FILE(LOG_AMP_HEX)) lar (
        .clk(clk), .addr(pixel_in), .dout(xc_lut)
    );
    // ROM read is registered (1 cycle); its own valid is pixel_in_valid delayed 1.
    reg pv_d1;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) pv_d1 <= 1'b0;
        else       pv_d1 <= pixel_in_valid;
    end

    // ---- Stage: power expansion (xc -> xc,xc2,xc3, aligned) ----------------
    wire pe_valid;
    wire signed [DW-1:0]  xc_pe;
    wire        [X2W-1:0] xc2_pe;
    wire signed [X3W-1:0] xc3_pe;
    power_expand #(.DW(DW), .X2W(X2W), .X3W(X3W)) pe (
        .clk(clk), .rstn(rstn), .valid_in(pv_d1), .xc_in(xc_lut),
        .valid_out(pe_valid), .xc_out(xc_pe), .xc2_out(xc2_pe), .xc3_out(xc3_pe)
    );

    // ---- Stage: 3 lock-stepped line buffers ---------------------------------
    wire wv_x, wv_x2, wv_x3;
    wire [SLI*SLI*DW-1:0]  win_x;
    wire [SLI*SLI*X2W-1:0] win_x2;
    wire [SLI*SLI*X3W-1:0] win_x3;

    line_buffer #(.SLI(SLI), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT), .DATA_WIDTH(DW)) lb_x (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pe_valid), .pixel_in(xc_pe),
        .window_valid(wv_x), .window_pixels(win_x)
    );
    line_buffer #(.SLI(SLI), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT), .DATA_WIDTH(X2W)) lb_x2 (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pe_valid), .pixel_in(xc2_pe),
        .window_valid(wv_x2), .window_pixels(win_x2)
    );
    line_buffer #(.SLI(SLI), .IMG_WIDTH(IMG_WIDTH), .IMG_HEIGHT(IMG_HEIGHT), .DATA_WIDTH(X3W)) lb_x3 (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pe_valid), .pixel_in(xc3_pe),
        .window_valid(wv_x3), .window_pixels(win_x3)
    );
    // wv_x/wv_x2/wv_x3 are structurally identical (same SLI/IMG_WIDTH control
    // path, only DATA_WIDTH differs, which does not affect the control path)
    // -- verified equal in simulation, used interchangeably below.

    // ---- Centre-pixel tap (zero-cost static bit-slice, see file header) ----
    localparam integer CTR_IDX    = TK_SLI*SLI + TK_SLI;
    localparam integer CTR_BITPOS = (SLI*SLI - 1 - CTR_IDX) * DW;
    wire signed [DW-1:0] x_centre = win_x[CTR_BITPOS +: DW];

    // ---- Stage: box sums (adds only, one instance per plane) ---------------
    wire s1v, s2v, s3v;
    wire signed [S1W-1:0] sum1_w, sum1_g;
    wire signed [S2W-1:0] sum2_w, sum2_g;
    wire signed [S3W-1:0] sum3_w, sum3_g;

    box_sum_inc #(.SLI(SLI), .GUARD(GUARD), .DATA_WIDTH(DW),  .SUM_WIDTH(S1W), .SIGNED_DATA(1)) bs1 (
        .clk(clk), .rstn(rstn), .window_valid(wv_x),  .window_plane(win_x),
        .sums_valid(s1v), .sum_window(sum1_w), .sum_guard(sum1_g)
    );
    box_sum_inc #(.SLI(SLI), .GUARD(GUARD), .DATA_WIDTH(X2W), .SUM_WIDTH(S2W), .SIGNED_DATA(0)) bs2 (
        .clk(clk), .rstn(rstn), .window_valid(wv_x2), .window_plane(win_x2),
        .sums_valid(s2v), .sum_window(sum2_w), .sum_guard(sum2_g)
    );
    box_sum_inc #(.SLI(SLI), .GUARD(GUARD), .DATA_WIDTH(X3W), .SUM_WIDTH(S3W), .SIGNED_DATA(1)) bs3 (
        .clk(clk), .rstn(rstn), .window_valid(wv_x3), .window_plane(win_x3),
        .sums_valid(s3v), .sum_window(sum3_w), .sum_guard(sum3_g)
    );

    // ---- Centre pixel, delayed to meet box_sum_inc's 1-cycle latency -------
    wire xc1v; wire signed [DW-1:0] x_ctr_d1;
    pipe_delay #(.DATA_WIDTH(DW), .DEPTH(1)) ctr_delay1 (
        .clk(clk), .rstn(rstn), .in_valid(wv_x), .in_data(x_centre),
        .out_valid(xc1v), .out_data(x_ctr_d1)
    );

    // ---- Stage: moment/cumulant pipeline (10 cycles) ------------------------
    molc_estimator3 #(.S1W(S1W), .S2W(S2W), .S3W(S3W), .C1W(C1W), .C2W(C2W), .C3W(C3W)) me3 (
        .clk(clk), .rstn(rstn), .sums_valid(s1v),
        .sum1_window(sum1_w), .sum1_guard(sum1_g),
        .sum2_window(sum2_w), .sum2_guard(sum2_g),
        .sum3_window(sum3_w), .sum3_guard(sum3_g),
        .molc_valid(molc_valid), .c1_code(c1_code), .c2_code(c2_code), .c3_code(c3_code)
    );

    // molc_estimator3 has 10 register stages from its inputs (s1v/sum*) to its
    // outputs -- matching that exactly, not by formula but by the SAME literal
    // stage count derived and commented in molc_estimator3.v itself, so a
    // future change to that pipeline's depth is caught here by a testbench
    // mismatch rather than silently going stale.
    localparam integer MOLC3_DEPTH = 10;
    pipe_delay #(.DATA_WIDTH(DW), .DEPTH(MOLC3_DEPTH)) ctr_delay2 (
        .clk(clk), .rstn(rstn), .in_valid(xc1v), .in_data(x_ctr_d1),
        .out_valid(), .out_data(x_code)
    );

endmodule
