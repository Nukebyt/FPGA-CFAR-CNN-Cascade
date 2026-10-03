// threshold_compare3.v
// Shared final stage for every detector: T_log = c1 + delta, detect = x > T_log.
//
// front_end3's x_code is CENTRED (X0 already subtracted, baked into
// log_amp_rom.v at generation time -- see fixedpoint_config.m), but c1_code
// is deliberately UNCENTRED (X0 added back in molc_estimator3's Stage 6, so
// c1 reads as a real log-amplitude for anyone inspecting it). Comparing a
// centred x against an uncentred T_log directly would be silently wrong by
// exactly X0 on every single decision. Rather than re-add X0 to x_code once
// per PIXEL (expensive: it's on the per-pixel critical path), this subtracts
// X0 from T_log once per WINDOW (delta and c1 already update at window rate)
// -- same result, cheaper.
module threshold_compare3 #(
    parameter integer C1W = 16, parameter integer C1_FRAC = 13,
    parameter integer DELTAW = 16, parameter integer DELTA_FRAC = 10,
    parameter integer XW = 16, parameter integer X_FRAC = 14,
    parameter integer TLOGW = 17, parameter integer TLOG_FRAC = 10,
    parameter signed [31:0] X0_CODE_TLOGFRAC = 1242 // round(1.2125*2^TLOG_FRAC)
) (
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     in_valid,
    input  wire signed [C1W-1:0]    c1_code,
    input  wire signed [DELTAW-1:0] delta_code,
    input  wire signed [XW-1:0]     x_code,       // centred, X_FRAC frac bits

    output reg                      detect_valid,
    output reg                      detect,
    output reg signed [TLOGW-1:0]   T_log_code    // uncentred, diagnostic
);
    // c1 (C1_FRAC) -> TLOG_FRAC, exact when TLOG_FRAC <= C1_FRAC (a right
    // shift with rounding); delta is already at TLOG_FRAC by construction
    // (every detector's delta format shares C.delta's frac bits).
    function automatic signed [63:0] rshift_round;
        input signed [63:0] v; input integer sh;
        reg signed [63:0] rb;
        begin
            rb = (sh > 0) ? (64'sd1 <<< (sh-1)) : 64'sd0;
            rshift_round = (v >= 0) ? ((v + rb) >>> sh) : -((-v + rb) >>> sh);
        end
    endfunction

    localparam integer SH_C1 = C1_FRAC - TLOG_FRAC;

    wire signed [63:0] c1_at_tlog = (SH_C1 >= 0)
        ? rshift_round(c1_code, SH_C1)
        : ($signed(c1_code) <<< (-SH_C1));

    wire signed [TLOGW-1:0] T_log_uncentred = c1_at_tlog[TLOGW-1:0] + delta_code;
    wire signed [TLOGW-1:0] T_log_centred   = T_log_uncentred - X0_CODE_TLOGFRAC[TLOGW-1:0];

    // Align T_log_centred (TLOG_FRAC) to x_code's own frac (X_FRAC) for the
    // compare -- an exact left shift when X_FRAC >= TLOG_FRAC (true for
    // every detector here: x is 14, T_log is 10).
    localparam integer SH_CMP = X_FRAC - TLOG_FRAC;
    wire signed [TLOGW+SH_CMP:0] T_log_cmp = $signed(T_log_centred) <<< SH_CMP;
    wire signed [TLOGW+SH_CMP:0] x_wide    = $signed(x_code);

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            detect_valid <= 1'b0; detect <= 1'b0; T_log_code <= 0;
        end else begin
            detect_valid <= in_valid;
            detect       <= (x_wide > T_log_cmp);
            T_log_code   <= T_log_uncentred;
        end
    end
endmodule
