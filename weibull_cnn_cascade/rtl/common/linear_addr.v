// linear_addr.v
// a = round(x*SCALE + OFFSET), clamped to [0, N_ENTRIES-1], with saturation
// flags -- the SAME formula every detector's address generation reduces to:
//   GenGamma/Burr : x = log2|s|             (via log2_lzc)
//   G0            : x = log2(c2), log2|dnorm| (two instances, via log2_lzc)
//   Weibull/Lognormal : x = c2 directly     (no log2_lzc needed)
// One module, reused five times with different upstream x and constants,
// rather than five near-duplicate address generators.
//
// SCALE_Q/OFFSET_Q are hardcoded fixed constants (per checklist 1.11 --
// never derived from a runtime division in RTL), computed by the
// corresponding dump-constants MATLAB step.
//
// *** SCALE BOOKKEEPING, GET THIS WRONG AND EVERY ADDRESS SATURATES ***
// x arrives as an INTEGER CODE, not a real number: x_code = x_real * 2^X_FRAC
// (e.g. log2_lzc's output has X_FRAC=16 fractional bits baked in). SCALE_Q is
// ALSO a scaled integer: SCALE_Q = round(scale_real * 2^SCALE_FRAC). Their
// product therefore carries X_FRAC+SCALE_FRAC fractional bits, not just
// SCALE_FRAC -- a first version of this module shifted by SCALE_FRAC alone
// (and scaled OFFSET_Q to 2^SCALE_FRAC to match), which for X_FRAC=16 made
// every address off by a factor of 2^16, saturating essentially every real
// input to address 0. Caught by gengamma_backend_tb.v producing a small,
// input-independent set of repeated "got" values (the delta table's OWN
// address-0 entry, looked up over and over) -- the signature of a stuck
// address, not scattered numeric noise.
module linear_addr #(
    parameter integer IN_WIDTH   = 22,  // x's total width (signed)
    parameter integer X_FRAC     = 16,  // x's OWN fractional bits (e.g. log2_lzc's OUT_FRAC)
    parameter integer SCALE_FRAC = 16,  // fractional bits baked into SCALE_Q/OFFSET_Q
    parameter integer ADDR_BITS  = 12,  // N_ENTRIES = 2^ADDR_BITS (4096 -> 12)
    parameter signed [47:0] SCALE_Q  = 0,                      // round(scale_real * 2^SCALE_FRAC)
    parameter signed [95:0] OFFSET_Q = 0                       // round(offset_real * 2^(X_FRAC+SCALE_FRAC))
) (
    input  wire                       clk,
    input  wire                       valid_in,
    input  wire signed [IN_WIDTH-1:0] x,

    output reg                        valid_out,
    output reg  [ADDR_BITS-1:0]       addr,
    output reg                        sat_low,
    output reg                        sat_high
);
    localparam integer N_ENTRIES = (1 << ADDR_BITS);
    localparam integer TOTAL_FRAC = X_FRAC + SCALE_FRAC;

    wire signed [IN_WIDTH+48:0] prod = x * SCALE_Q;                 // frac = TOTAL_FRAC
    wire signed [IN_WIDTH+96:0] sum_aligned = prod + OFFSET_Q;      // frac = TOTAL_FRAC

    wire signed [IN_WIDTH+96:0] rounded = (sum_aligned >= 0)
        ? ((sum_aligned + (1 <<< (TOTAL_FRAC-1))) >>> TOTAL_FRAC)
        : -((-sum_aligned + (1 <<< (TOTAL_FRAC-1))) >>> TOTAL_FRAC);

    reg sl_c, sh_c;
    reg [ADDR_BITS-1:0] addr_c;
    always @(*) begin
        if (rounded < 0) begin
            sl_c = 1'b1; sh_c = 1'b0; addr_c = {ADDR_BITS{1'b0}};
        end else if (rounded > N_ENTRIES-1) begin
            sl_c = 1'b0; sh_c = 1'b1; addr_c = N_ENTRIES-1;
        end else begin
            sl_c = 1'b0; sh_c = 1'b0; addr_c = rounded[ADDR_BITS-1:0];
        end
    end

    always @(posedge clk) begin
        valid_out <= valid_in;
        addr      <= addr_c;
        sat_low   <= sl_c;
        sat_high  <= sh_c;
    end
endmodule
