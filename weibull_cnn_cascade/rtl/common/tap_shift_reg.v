// tap_shift_reg.v
// A DEPTH-deep, DATA_WIDTH-wide shift register exposing ALL taps
// simultaneously (unlike pipe_delay.v, which only exposes the deepest one).
// Same flat-bit-vector concatenation idiom as pipe_delay.v -- Quartus's
// standard shift-register (altshift_taps-style) inference target, already
// benchmarked fast in this project (0.178s for 2000 cycles at
// DATA_WIDTH=16384/DEPTH=10 in isolation) and, per pipe_delay.v's own
// header, the documented way to get Quartus to map a long chain onto
// on-chip memory instead of DEPTH discrete flip-flops.
//
// WHY THIS EXISTS: line_buffer.v's column-shift storage (col_shift, a raw
// 2-D `reg [DATA_WIDTH-1:0] col_shift[0:SLI-1][0:SLI-1]` array, one SLI-deep
// shift register per row-tap) is real flip-flops, not RAM -- costing
// SLI*SLI*DATA_WIDTH bits of flip-flops PER PLANE. At SLI=17 (Weibull's own
// verified production geometry) this is already ~89% of that detector's
// entire measured register count; at SLI=41 (the smallest geometry that
// geometrically fits a 32x32 CNN patch tap -- see front_end3_patchtap.v)
// the projected cost (~118k registers just for this array) is well past
// this chip's real capacity -- confirmed empirically, not just estimated:
// quartus_map on the unmodified line_buffer.v at SLI=41 did not finish
// register optimization in 50+ minutes and 1.2GB of memory, an unambiguous
// real-world signal, not merely a simulation-speed nuisance.
//
// This module replaces ONE row-tap's col_shift[k][0:SLI-1] with a single
// SLI-deep shift register of the same total bit content, letting Quartus
// infer memory (M10K shift-register mode) instead of raw flip-flops for it
// -- verified via a direct resource-probe comparison against the original
// line_buffer.v (see line_buffer_v2_tb.v and the quartus/probe/ project),
// not assumed from this reasoning alone.
//
// tap[0] = newest (this cycle's incoming value once committed on the next
// edge) .. tap[DEPTH-1] = oldest (DEPTH-1 valid-cycles ago) -- identical
// semantics and read-timing to col_shift[k][0]..col_shift[k][SLI-1] in the
// original line_buffer.v (a continuous read of a nonblocking-updated reg,
// so on any given cycle it reflects the state committed on the PREVIOUS
// edge, before this cycle's new sample is absorbed -- same one-cycle-latent
// convention the original design already used).
module tap_shift_reg #(
    parameter integer DATA_WIDTH = 16,
    parameter integer DEPTH = 41
) (
    input  wire                          clk,
    input  wire                          rstn,
    input  wire                          in_valid,
    input  wire signed [DATA_WIDTH-1:0]  in_data,

    output wire signed [DATA_WIDTH*DEPTH-1:0] taps  // flattened, tap 0 MSB-most (see file header + tap_at below)
);
    reg signed [DATA_WIDTH*DEPTH-1:0] data_sr;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            data_sr <= {(DATA_WIDTH*DEPTH){1'b0}};
        end else if (in_valid) begin
            data_sr <= {data_sr[DATA_WIDTH*(DEPTH-1)-1:0], in_data};
        end
    end

    assign taps = data_sr;

endmodule
