// power_expand.v
// ---------------------------------------------------------------------------
// Computes xc^2 and xc^3 from each incoming centred log-amplitude sample,
// ONCE per pixel as it enters the pipeline -- not recomputed later for both
// the window's entering AND leaving edge, which is what the older
// (2-moment-only) window_sum.v does internally. Computing each power once
// here and letting it flow through line_buffer (3 parallel instances, one per
// plane: x, x2, x3) means eviction reads an ALREADY-COMPUTED delayed value
// instead of recomputing it -- this is what makes the measured DSP law
// DSP = (SLI+GUARD)*(MOMENTS-1) (see rtl/probe/) come out at SLI+GUARD
// multipliers per moment instead of 2*(SLI+GUARD).
//
// Matches cfar_front_end_fixed.m Stage 2 exactly:
//   xc2_code = round(xc_code^2 / 2^(2*xc.frac - xc2.frac))
//   xc3_code = round(xc_code^3 / 2^(3*xc.frac - xc3.frac))
//
// Pipeline (checklist 1.10 -- one multiply per stage, never chained):
//   stage 0: register xc in
//   stage 1: SQUARE  (1 multiply)  -> xc2, and xc delayed to meet it
//   stage 2: CUBE    (1 multiply, reusing xc2) -> xc3, and xc/xc2 delayed
//            to emerge together on the same cycle
//
// Output xc_out/xc2_out/xc3_out/valid_out all arrive on the SAME cycle, so
// the three line_buffer instances downstream can be driven in lockstep.
// ---------------------------------------------------------------------------

module power_expand #(
    parameter integer DW   = 16,  // xc width, Q1.14 signed
    parameter integer X2W  = 26,  // xc2 width, Q2.24 unsigned
    parameter integer X3W  = 27,  // xc3 width, Q2.24 signed (int=2,frac=24,sign)
    parameter integer SH2  = 4,   // xc^2 raw frac(28) -> xc2.frac(24)
    // *** 14, NOT 18 -- THIS MODULE REUSES THE RESCALED xc2, MATLAB DOES NOT ***
    // cfar_front_end_fixed.m computes xc3_code = round(xc_code^3 / 2^18),
    // cubing the RAW 14-frac xc directly (42 raw frac bits, since 3*14=42).
    // This module instead reuses the ALREADY-RESCALED xc2 (24 frac bits, per
    // SH2 above) to save a multiplier -- exactly the reuse rtl/probe/
    // moment_probe.v measured the DSP cost of (see its own SH3=14, not 18).
    // Q2.24 xc2 (24 frac) times Q1.14 xc (14 frac) is 38 raw frac bits, so
    // reaching xc3.frac=24 needs a shift of 38-24=14, not 42-24=18. Using 18
    // here selects bits [18 +: 27] = [44:18] out of a 43-bit product ([42:0])
    // -- bits 43/44 do not exist, reading them returns X, and X contaminates
    // every downstream arithmetic op that touches xc3 (S3, prod3_1, s3n_2,
    // ... all the way to c3_code). Caught by front_end3_tb.v reporting c3
    // as literal 'x' on every window, not by any width-overflow warning at
    // elaboration time -- Verilog part-selects past a vector's bound are
    // silently legal and simply read as unknown.
    //
    // The consequence of the reuse (accepted, not a bug): xc3 here carries
    // ONE extra rounding step (xc2's SH2 rounding, then xc3's own SH3
    // rounding) that cfar_front_end_fixed.m's direct cube does not, so RTL
    // and the MATLAB golden model are no longer expected to be bit-identical
    // on c3/c3-derived quantities -- only close, at a magnitude the
    // per-detector gate's tolerance must be re-checked against, not assumed.
    parameter integer SH3  = 14
) (
    input  wire                  clk,
    input  wire                  rstn,
    input  wire                  valid_in,
    input  wire signed [DW-1:0]  xc_in,

    output reg                   valid_out,
    output reg signed [DW-1:0]   xc_out,
    output reg        [X2W-1:0]  xc2_out,
    output reg signed [X3W-1:0]  xc3_out
);

    localparam integer ROUND2 = (SH2 > 0) ? (1 << (SH2-1)) : 0;
    localparam integer ROUND3 = (SH3 > 0) ? (1 << (SH3-1)) : 0;

    // ---- Stage 0: register the input --------------------------------------
    reg signed [DW-1:0] xc_s0;
    reg                 v_s0;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            xc_s0 <= {DW{1'b0}};
            v_s0  <= 1'b0;
        end else begin
            xc_s0 <= xc_in;
            v_s0  <= valid_in;
        end
    end

    // ---- Stage 1: SQUARE (one multiply) -----------------------------------
    // Full product is 2*DW bits, always non-negative (a square). Round with
    // a +half-LSB add before the shift (round-half-away-from-zero on a
    // non-negative value == round-half-up, matching MATLAB's round()).
    wire signed [2*DW-1:0] prod_sq = xc_s0 * xc_s0;
    wire        [2*DW-1:0] prod_sq_rnd = prod_sq + ROUND2;

    reg        [X2W-1:0] xc2_s1;
    reg signed [DW-1:0]  xc_s1;
    reg                  v_s1;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            xc2_s1 <= {X2W{1'b0}};
            xc_s1  <= {DW{1'b0}};
            v_s1   <= 1'b0;
        end else begin
            xc2_s1 <= prod_sq_rnd[SH2 +: X2W];
            xc_s1  <= xc_s0;
            v_s1   <= v_s0;
        end
    end

    // ---- Stage 2: CUBE (a SECOND multiply, its OWN stage) ------------------
    // Reuses xc2 (already computed) rather than recomputing xc*xc*xc: one
    // extra multiply, not two. xc2 is unsigned; zero-extend by one bit so the
    // product with signed xc is a clean signed multiply.
    wire signed [X2W:0]         xc2_s   = {1'b0, xc2_s1};
    wire signed [X2W+DW:0]      prod_cu = xc2_s * xc_s1;
    wire signed [X2W+DW:0]      prod_cu_rnd =
        (prod_cu >= 0) ? (prod_cu + ROUND3) : (prod_cu - ROUND3);

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            valid_out <= 1'b0;
            xc_out    <= {DW{1'b0}};
            xc2_out   <= {X2W{1'b0}};
            xc3_out   <= {X3W{1'b0}};
        end else begin
            valid_out <= v_s1;
            xc_out    <= xc_s1;
            xc2_out   <= xc2_s1;
            xc3_out   <= prod_cu_rnd[SH3 +: X3W];
        end
    end

endmodule
