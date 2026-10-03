// sqrt_lzc.v
// sqrt of a non-negative fixed-point CODE, by leading-zero-count + mantissa
// ROM -- the exact construction _common/sqrt_fixed.m models:
//     normalise code = m * 2^(2j), m in [1,4)   (barrel shift by an EVEN
//                                                 amount, so the exponent
//                                                 halves exactly)
//     sqrt(code) = sqrt(m) * 2^j                 (small ROM + shift)
//     sqrt(value) = sqrt(code) * 2^(-fracBits/2)
//
// Unlike log2_lzc (a fixed left-normalise + one add), this needs a genuine
// bidirectional variable shift at the end: 2^j can move the mantissa's value
// up OR down depending on the input's own magnitude. Implemented as one
// signed variable-amount shift (<<< / >>> with a runtime-computed signed
// amount) -- costs ALMs for the barrel shifter, no DSP.
//
// code==0 clears `valid` and returns 0, matching sqrt_fixed.m.
module sqrt_lzc_b #(
    parameter integer WIDTH     = 26,  // code width (magnitude, unsigned)
    parameter integer FRACBITS  = 24,  // code's own fractional bits -- MUST be even
    parameter integer MANT_BITS = 10,  // mantissa ROM address bits (1024 entries)
    parameter integer OUT_FRAC  = 14,  // output format: unsigned Q(OUT_INT).OUT_FRAC
    parameter integer OUT_INT   = 2,
    parameter MANT_HEX = "lut/shared/sqrt_mant_lut.hex"
) (
    input  wire                       clk,
    input  wire [WIDTH-1:0]           code,
    output reg                        valid,
    output reg  [OUT_INT+OUT_FRAC-1:0] sq
);
    initial begin
        if (FRACBITS % 2 != 0)
            $fatal(1, "sqrt_lzc: FRACBITS must be even (got %0d)", FRACBITS);
    end

    localparam integer MANT_FRAC = 14; // sqrt_mant_lut.mat format (Q1.14 unsigned)
    localparam integer EW = $clog2(WIDTH);
    localparam integer OUTW = OUT_INT + OUT_FRAC;

    reg [14:0] mant_rom [0:(1<<MANT_BITS)-1]; // sqrt_mant_lut values are Q1.14 (15 bits, top bit can be 1: max 1.99994)
    initial $readmemh(MANT_HEX, mant_rom);

    // ---- combinational: priority encoder + force-even + normalise ----------
    integer k;
    reg                 found_c;
    reg signed [EW:0]   e_c;        // signed: exponent can need one extra bit
    reg [WIDTH-1:0]     shifted_c;
    // (m-1) at a common scale of 2^(WIDTH-1): ranges [0,2^(WIDTH-1)) in the
    // even-exponent case but up to just under 3*2^(WIDTH-1) in the odd case
    // (m up to just under 4 there, not 2) -- needs 2 bits MORE than
    // shifted_c's own width, not the same width. A first version declared
    // this at WIDTH-2 bits (matching only the even case) and silently
    // truncated the odd case's larger range.
    reg [WIDTH:0]       frac_norm_c;
    reg [MANT_BITS-1:0] mant_addr_c;
    reg [WIDTH+MANT_BITS+2:0] mant_scaled_c;

    reg signed [EW:0] e_true_c; // TRUE MSB position, before forcing even

    always @(*) begin
        found_c = 1'b0;
        e_true_c = 0;
        for (k = WIDTH-1; k >= 0; k = k-1) begin
            if (!found_c && code[k]) begin
                e_true_c = k[EW:0];
                found_c = 1'b1;
            end
        end
        // e_c (forced even, rounded DOWN) is the value used for the FINAL
        // 2^(e/2) exponent; the SHIFT AMOUNT below must stay based on the
        // TRUE (unadjusted) MSB position, not e_c -- shifting by e_c's
        // amount when e_true_c is odd pushes the true MSB one bit past the
        // top of a WIDTH-bit register (silently lost, not flagged), corrupting
        // the mantissa for every input whose MSB sits at an odd bit position
        // (~half of all inputs) while leaving even-MSB inputs looking correct.
        // Caught only by sweeping magnitudes across many bit-lengths in
        // lzc_tb.v, not by a handful of hand-picked values.
        e_c = e_true_c[0] ? (e_true_c - 1) : e_true_c;
        shifted_c = found_c ? (code << (WIDTH-1-e_true_c)) : {WIDTH{1'b0}};
        // shifted_c always represents m1 = code/2^e_true_c in [1,2), as
        // Q1.(WIDTH-1) (leading 1 implicit at bit WIDTH-1). m = m1*2^adj,
        // adj = e_true_c-e_c in {0,1}. Normalise (m-1) to a COMMON scale of
        // 2^(WIDTH-1) regardless of adj, so the same n/(3*2^(WIDTH-1))
        // constant multiply below serves both parities:
        //   adj=0: m=m1,   (m-1)*2^(WIDTH-1) = shifted_c[WIDTH-2:0]
        //   adj=1: m=2*m1, (m-1)*2^(WIDTH-1) = (shifted_c - 2^(WIDTH-2)) << 1
        if (!e_true_c[0])
            frac_norm_c = {1'b0, shifted_c[WIDTH-2:0]};
        else
            frac_norm_c = (shifted_c - (1 <<< (WIDTH-2))) << 1;
        // mantissa m in [1,4): the fractional part for addressing is
        // (m-1)/3 -- see sqrt_fixed.m's `(m-1)/3*n`. Realised as a multiply
        // by a compile-time constant round(n/3 * 2^K)/2^K; K chosen so the
        // rounding error is negligible next to the table's own 1/n step.
        mant_scaled_c = ({{MANT_BITS{1'b0}}, frac_norm_c}
                          * ((1<<MANT_BITS) * 32'd699051 >> 21)) // *(n/3), 699051/2^21 ~= 1/3.0000002
                         + (1 <<< (WIDTH-2));
        mant_addr_c = mant_scaled_c >> (WIDTH-1);
    end

    // ---- stage 1 register ----------------------------------------------------
    reg               found_1;
    reg signed [EW:0] e_1;
    reg [MANT_BITS-1:0] mant_addr_1;
    always @(posedge clk) begin
        found_1     <= found_c;
        e_1         <= e_c;
        mant_addr_1 <= mant_addr_c;
    end

    // ---- stage 2: ROM read, then the variable shift + final fracBits scale -
    wire [14:0] mant_val = mant_rom[mant_addr_1]; // Q1.14 unsigned, value in [1,2)... actually [1,~2) per table
    // total_shift = e_1/2 - FRACBITS/2, applied to a Q1.14 mantissa to reach
    // OUT_FRAC fractional bits: net left-shift amount =
    //   (e_1/2 - FRACBITS/2) + (OUT_FRAC - MANT_FRAC)
    wire signed [15:0] half_e = e_1 >>> 1; // arithmetic shift; e_1 forced even so exact
    wire signed [15:0] net_shift = half_e - (FRACBITS/2) + (OUT_FRAC - MANT_FRAC);
    wire signed [OUTW+16:0] mant_wide = $signed({1'b0, mant_val});
    // Right-shift branch needs an explicit round-half-up (add half an LSB of
    // the shift-out BEFORE truncating) -- a first version used a plain `>>>`
    // here, which always rounds toward -infinity for a non-negative value
    // (i.e. truncates), producing a small but consistent NEGATIVE bias on
    // every input landing in this branch. Caught by lzc_tb.v's relative-
    // tolerance check failing on the clear majority of sqrt vectors with
    // `got` always slightly below `expected` -- a uniform sign on the error
    // is the signature of a missing round, not table-quantisation noise
    // (which would scatter both directions).
    wire signed [OUTW+16:0] round_bit = (net_shift < 0) ? (1 <<< (-net_shift-1)) : {(OUTW+17){1'b0}};
    wire signed [OUTW+16:0] shifted_result =
        (net_shift >= 0) ? (mant_wide <<< net_shift) : ((mant_wide + round_bit) >>> (-net_shift));

    always @(posedge clk) begin
        valid <= found_1;
        sq    <= shifted_result[OUTW-1:0];
    end
endmodule
