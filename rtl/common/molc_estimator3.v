// molc_estimator3.v
// ---------------------------------------------------------------------------
// Stage 3/4/5/6 of cfar_front_end_fixed.m: turns the three raw power sums
// (S1,S2,S3 -- already window-minus-guard) into c1, c2, c3, by reciprocal-
// multiply (never a runtime divider) and the cancellation-sensitive moment
// formulas
//     m1 = S1/N
//     m2 = S2/N - m1^2                    (c2 after the unbiased-N scale)
//     m3 = S3/N - 3*m1*(S2/N) + 2*m1^3    (c3 after the unbiased-N scale)
//
// UNLIKE the existing 2-moment molc_estimator.v (which does three multiplies
// back-to-back in ONE always block -- a genuine chained-multiply timing risk
// per checklist 1.10, and plausibly the source of the "three-clock-period
// critical path" the moment_probe.v header describes from this project's own
// history), every multiply here gets its OWN pipeline stage, registered
// before the next one starts. Nine stages total; throughput is one window's
// c1/c2/c3 per cycle regardless (the pipeline is fully streaming), latency is
// nine cycles.
//
// Constants (INV_N, INV_NM1, K2_MULT, K3_MULT, X0_CODE, and every shift
// amount) are computed by sarfish_sample/dump_rtl_constants.m from the LIVE
// fixedpoint_config.m (sli=17/guard=13, RECIP_M=28) and HARDCODED as module
// parameters below -- never computed in RTL from (1<<M)/N, per checklist
// 1.11 and BUG_LOG H5 (Verilog integer division truncates; MATLAB's round()
// does not, and the two diverge on most real geometries).
//
// Formats (fixedpoint_config.m, sli=17/guard=13):
//   S1  Q8.14  signed  23b     S2  Q9.24 unsigned 33b     S3  Q9.24 signed 34b
//   m1  Q1.24  signed  26b     c1  Q2.13  signed   16b
//   c2  Q2.24 unsigned 26b     c3  Q3.24  signed   28b
// ---------------------------------------------------------------------------

module molc_estimator3 #(
    parameter integer S1W = 23, parameter integer S2W = 33, parameter integer S3W = 34,
    parameter integer M1W = 26, parameter integer C1W = 16, parameter integer C2W = 26, parameter integer C3W = 28,
    parameter integer XC_FRAC = 14, parameter integer M1_FRAC = 24,
    parameter integer XC2_FRAC = 24, parameter integer XC3_FRAC = 24,
    parameter integer C1_FRAC = 13, parameter integer C2_FRAC = 24, parameter integer C3_FRAC = 24,
    // ---- hardcoded, audited constants (dump_rtl_constants.m) --------------
    parameter integer RM        = 28,          // RECIP_M
    parameter integer INV_N     = 2236962,
    parameter integer INV_NM1   = 2255760,
    parameter integer K2_MULT   = 270691216,
    parameter integer K3_MULT   = 275279203,
    parameter integer X0_CODE_M1FRAC = 20342374,
    // derived shifts (also from dump_rtl_constants.m)
    parameter integer SH_M1   = 18,
    parameter integer SH_S2N  = 28,
    parameter integer SH_S3N  = 28,
    parameter integer SH_M1SQ = 24,
    parameter integer SH_M1S2 = 24,
    parameter integer SH_M1CU = 48,
    parameter integer SH_K2   = 28,
    parameter integer SH_K3   = 28,
    parameter integer SH_C1   = 11
) (
    input  wire                      clk,
    input  wire                      rstn,
    input  wire                      sums_valid,
    input  wire signed [S1W-1:0]     sum1_window, sum1_guard,
    input  wire signed [S2W-1:0]     sum2_window, sum2_guard,   // unsigned magnitude carried in a signed reg
    input  wire signed [S3W-1:0]     sum3_window, sum3_guard,

    output reg                       molc_valid,
    output reg signed [C1W-1:0]      c1_code,
    output reg        [C2W-1:0]      c2_code,   // c2 >= 0 always (floored, see Stage6)
    output reg signed [C3W-1:0]      c3_code
);

    // Widened to 96 bits, not 64: m1^3's raw product (m1sq_raw * m1, each up
    // to ~2^25/~2^50 in magnitude for a bright/dark near-uniform window --
    // exactly fixedpoint_config.m's "m1 IS THE ONE THAT ACTUALLY MATTERED"
    // case, BUG_LOG D8) needs ~75 bits. A first version of this function used
    // 64 bits throughout; it passed on small hand-picked test sums (isolated
    // debug_c3.v) and produced silent X/wraparound on real image data the
    // moment a window's mean sat far from the centring constant X0 -- caught
    // only by simulating against real cfar_front_end_fixed.m vectors, not by
    // the isolated unit probe. Every OTHER product in this pipeline (S*INV_N,
    // m1*s2n, m2*K2_MULT, m3*K3_MULT) stays under 60 bits and would fit in
    // 64, but the function is shared, so it is sized for the worst caller.
    function automatic signed [95:0] rshift_round;
        // round-half-away-from-zero, then arithmetic shift -- matches
        // MATLAB round(v/2^sh) exactly (BUG_LOG H5 convention).
        input signed [95:0] v;
        input integer sh;
        reg signed [95:0] rb;
        begin
            rb = (sh > 0) ? (96'sd1 <<< (sh-1)) : 96'sd0;
            rshift_round = (v >= 0) ? ((v + rb) >>> sh) : -((-v + rb) >>> sh);
        end
    endfunction

    // ==== Stage 0: window - guard (subtract only) ==========================
    reg signed [S1W-1:0] S1_0; reg signed [S2W-1:0] S2_0; reg signed [S3W-1:0] S3_0;
    reg v0;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            S1_0<=0; S2_0<=0; S3_0<=0; v0<=1'b0;
        end else begin
            S1_0 <= sum1_window - sum1_guard;
            S2_0 <= sum2_window - sum2_guard;
            S3_0 <= sum3_window - sum3_guard;
            v0   <= sums_valid;
        end
    end

    // ==== Stage 1: three parallel multiplies (S*INV_N) ======================
    // These are independent (no data dependency between them), so they share
    // one pipeline stage without chaining -- each is still exactly one
    // multiply deep.
    reg signed [63:0] prod1_1, prod2_1, prod3_1;
    reg v1;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin prod1_1<=0; prod2_1<=0; prod3_1<=0; v1<=1'b0; end
        else begin
            prod1_1 <= S1_0 * $signed({1'b0, INV_N[31:0]});
            prod2_1 <= S2_0 * $signed({1'b0, INV_N[31:0]});
            prod3_1 <= S3_0 * $signed({1'b0, INV_N[31:0]});
            v1 <= v0;
        end
    end

    // ==== Stage 2: round-shift each into its target format ==================
    reg signed [M1W-1:0] m1_2;
    reg signed [63:0]    s2n_2, s3n_2;
    reg v2;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin m1_2<=0; s2n_2<=0; s3n_2<=0; v2<=1'b0; end
        else begin
            m1_2  <= rshift_round(prod1_1, SH_M1);
            s2n_2 <= rshift_round(prod2_1, SH_S2N);   // xc2.frac scale
            s3n_2 <= rshift_round(prod3_1, SH_S3N);   // xc3.frac scale
            v2 <= v1;
        end
    end

    // ==== Stage 3: m1^2 (raw, full precision -- needed later for m1^3 too) =
    reg signed [95:0] m1sq_raw_3;
    reg signed [M1W-1:0] m1_3;
    reg signed [63:0] s2n_3, s3n_3;
    reg v3;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin m1sq_raw_3<=0; m1_3<=0; s2n_3<=0; s3n_3<=0; v3<=1'b0; end
        else begin
            m1sq_raw_3 <= m1_2 * m1_2;      // Q2.48 (2*M1_FRAC frac bits), always >=0
            m1_3  <= m1_2;
            s2n_3 <= s2n_2;
            s3n_3 <= s3n_2;
            v3 <= v2;
        end
    end

    // ==== Stage 4: m1sq rescaled to xc2.frac ; m1*s2n (raw) =================
    reg signed [63:0] m1sq_4;      // at xc2.frac
    reg signed [63:0] m1s2_raw_4;
    reg signed [95:0] m1sq_raw_4;  // carried forward, full precision, for the cube
    reg signed [M1W-1:0] m1_4;
    reg signed [63:0] s2n_4, s3n_4;
    reg v4;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            m1sq_4<=0; m1s2_raw_4<=0; m1sq_raw_4<=0; m1_4<=0; s2n_4<=0; s3n_4<=0; v4<=1'b0;
        end else begin
            m1sq_4     <= rshift_round(m1sq_raw_3, SH_M1SQ);
            m1s2_raw_4 <= m1_3 * s2n_3;
            m1sq_raw_4 <= m1sq_raw_3;
            m1_4  <= m1_3;
            s2n_4 <= s2n_3;
            s3n_4 <= s3n_3;
            v4 <= v3;
        end
    end

    // ==== Stage 5: m2 = s2n - m1sq ; m1s2 rescaled ; m1^3 (raw) =============
    reg signed [63:0] m2_5;          // xc2.frac
    reg signed [63:0] m1s2_5;        // xc3.frac
    reg signed [95:0] m1cu_raw_5;    // Q4.72-ish, full precision
    reg signed [63:0] s3n_5;
    reg v5;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin m2_5<=0; m1s2_5<=0; m1cu_raw_5<=0; s3n_5<=0; v5<=1'b0; end
        else begin
            m2_5       <= s2n_4 - m1sq_4;
            m1s2_5     <= rshift_round(m1s2_raw_4, SH_M1S2);
            m1cu_raw_5 <= m1sq_raw_4 * m1_4;
            s3n_5      <= s3n_4;
            v5 <= v4;
        end
    end

    // ==== Stage 6: m1cu rescaled ; m3 = s3n - 3*m1s2 + 2*m1cu ===============
    reg signed [63:0] m2_6, m3_6;
    reg v6;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin m2_6<=0; m3_6<=0; v6<=1'b0; end
        else begin
            m2_6 <= m2_5;
            m3_6 <= s3n_5 - 3*m1s2_5 + 2*rshift_round(m1cu_raw_5, SH_M1CU);
            v6 <= v5;
        end
    end

    // ==== Stage 7: unbiased-N scale (k2_mult, k3_mult), one mult each =======
    reg signed [63:0] c2raw_7, c3raw_7;
    reg v7;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin c2raw_7<=0; c3raw_7<=0; v7<=1'b0; end
        else begin
            c2raw_7 <= m2_6 * $signed({1'b0, K2_MULT[31:0]});
            c3raw_7 <= m3_6 * $signed({1'b0, K3_MULT[31:0]});
            v7 <= v6;
        end
    end

    // ==== Stage 8: round-shift into c2/c3 native format, floor c2 at 0 ======
    reg signed [63:0] c2_8, c3_8;
    reg v8;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin c2_8<=0; c3_8<=0; v8<=1'b0; end
        else begin
            c2_8 <= rshift_round(rshift_round(c2raw_7, SH_K2), C2_FRAC-XC2_FRAC); // usually 0 shift
            c3_8 <= rshift_round(rshift_round(c3raw_7, SH_K3), C3_FRAC-XC3_FRAC);
            v8 <= v7;
        end
    end

    // ==== Stage 9: c1 = m1 + X0 (rescaled) ; final outputs ==================
    // c1 only needs m1 -- carried alongside since stage 3 as m1_3/m1_4, but
    // by now it is several stages stale relative to c2/c3's OWN pipeline
    // depth. To keep c1/c2/c3 emerging on the SAME cycle (as one molc_valid
    // window), m1 is re-delayed by a plain shift register matching the extra
    // depth stages 3->8 add (5 cycles) rather than threaded stage-by-stage
    // through logic it does not need -- see the delay chain below.
    // Traced cycle-by-cycle against the c2/c3 pipeline (S2_0->prod2_1->s2n_2
    // ->s2n_3->s2n_4->m2_5->m2_6->c2raw_7->c2_8->c2_code, 10 register stages
    // from the input subtract to the output register): m1_4 settles at the
    // same absolute cycle as s2n_4 (both are 5 stages from the input), and
    // c2/c3 still pass through 5 MORE registers (m2_5,m2_6,c2raw_7,c2_8,
    // c2_code) before reaching the output. m1 must cross the same 5, but it
    // has no more real arithmetic to do -- so it is 4 plain delay registers
    // (m1_dly[0..3]) plus the shared final output register (c1_code) = 5.
    // Getting this wrong misaligns c1 from c2/c3 by whole cycles, silently
    // pairing one window's c1 with a DIFFERENT window's c2/c3 -- verified by
    // the testbench comparing c1/c2/c3 as a triple, not by inspection.
    localparam integer M1_DELAY = 4;
    reg signed [M1W-1:0] m1_dly [0:M1_DELAY-1];
    integer di;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (di = 0; di < M1_DELAY; di = di + 1) m1_dly[di] <= 0;
        end else begin
            m1_dly[0] <= m1_4;
            for (di = 1; di < M1_DELAY; di = di + 1) m1_dly[di] <= m1_dly[di-1];
        end
    end

    wire signed [M1W-1:0] m1_final = m1_dly[M1_DELAY-1];
    wire signed [63:0] c1_m1frac = $signed({{(64-M1W){m1_final[M1W-1]}}, m1_final}) + X0_CODE_M1FRAC;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            molc_valid <= 1'b0; c1_code <= 0; c2_code <= 0; c3_code <= 0;
        end else begin
            molc_valid <= v8;
            c1_code <= rshift_round(c1_m1frac, SH_C1);
            c2_code <= (c2_8 < 0) ? {C2W{1'b0}} : c2_8[C2W-1:0];   // floor at 0, Stage6 note
            c3_code <= c3_8[C3W-1:0];
        end
    end

endmodule
