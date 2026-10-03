// floor_addr.v
// Interpolation-table address generator: unlike linear_addr.v (which ROUNDS
// to the nearest single entry), this produces {index, weight} = floor(a) and
// its fractional remainder, for bilinear/linear INTERPOLATION -- G0's two
// address axes both need this, matching G0CFAR_FixedPoint.m's
//     f = (x-lo)/(hi-lo)*(N-1);  f = clamp(f, 0, N-1-eps);  i=floor(f); w=f-i;
// The clamp ceilings at N-2 (not N-1), so `index+1` is always a valid entry
// -- callers never need their own extra bounds check on the "+1" read.
//
// Same scale bookkeeping as linear_addr.v (D15): x is a CODE with X_FRAC
// fractional bits, SCALE_Q carries SCALE_FRAC fractional bits, so their
// product carries X_FRAC+SCALE_FRAC bits -- OFFSET_Q must be pre-scaled to
// that SAME combined scale.
module floor_addr #(
    parameter integer IN_WIDTH   = 22,
    parameter integer X_FRAC     = 18,
    parameter integer SCALE_FRAC = 16,
    parameter integer ADDR_BITS  = 6,   // N_ENTRIES = 2^ADDR_BITS (64 -> 6)
    parameter integer WEIGHT_BITS = 12,
    parameter signed [47:0] SCALE_Q  = 0,
    parameter signed [95:0] OFFSET_Q = 0
) (
    input  wire                       clk,
    input  wire                       valid_in,
    input  wire signed [IN_WIDTH-1:0] x,

    output reg                        valid_out,
    output reg  [ADDR_BITS-1:0]       index,
    output reg  [WEIGHT_BITS-1:0]     weight,
    output reg                        sat_low,
    output reg                        sat_high
);
    localparam integer N_ENTRIES = (1 << ADDR_BITS);
    localparam integer TOTAL_FRAC = X_FRAC + SCALE_FRAC;
    // clamp ceiling: just under (N_ENTRIES-1), at TOTAL_FRAC scale
    localparam signed [IN_WIDTH+96:0] MAX_RAW = ((N_ENTRIES-1) <<< TOTAL_FRAC) - 1;

    wire signed [IN_WIDTH+48:0] prod = x * SCALE_Q;
    wire signed [IN_WIDTH+96:0] sum_aligned = prod + OFFSET_Q;

    reg sl_c, sh_c;
    reg [ADDR_BITS-1:0] index_c;
    reg [WEIGHT_BITS-1:0] weight_c;
    reg signed [IN_WIDTH+96:0] clamped;
    always @(*) begin
        if (sum_aligned < 0) begin
            sl_c = 1'b1; sh_c = 1'b0; clamped = 0;
        end else if (sum_aligned > MAX_RAW) begin
            sl_c = 1'b0; sh_c = 1'b1; clamped = MAX_RAW;
        end else begin
            sl_c = 1'b0; sh_c = 1'b0; clamped = sum_aligned;
        end
        index_c  = clamped[TOTAL_FRAC +: ADDR_BITS];
        weight_c = clamped[TOTAL_FRAC-1 -: WEIGHT_BITS];
    end

    always @(posedge clk) begin
        valid_out <= valid_in;
        index     <= index_c;
        weight    <= weight_c;
        sat_low   <= sl_c;
        sat_high  <= sh_c;
    end
endmodule
