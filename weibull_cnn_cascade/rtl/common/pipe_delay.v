// pipe_delay.v
// Generic fixed-depth delay line: delays {valid, data} by DEPTH clock cycles.
// A plain shift-register chain (DEPTH single-cycle stages in series) -- simple
// to reason about, and Quartus can map a long chain like this onto on-chip
// memory (shift-register inference) rather than DEPTH discrete registers.
//
// Needed because threshold_compare.v's inputs do NOT arrive on the same cycle:
//   - c1_code (from molc_estimator) is ready 2 cycles before kc_code (from
//     kc_lut, chained through c2_addr_gen) for the SAME window.
//   - x_code, the cell-under-test, is the window's CENTRE pixel -- it streamed
//     in tk_sli rows and tk_sli columns BEFORE the window's bottom-right
//     corner closed the window, i.e. tk_sli*(IMG_WIDTH+1) cycles earlier, not
//     a small constant. This is not a bug to fix; it's the real cost of a
//     line-buffer architecture; the RTL must hold that pixel in a delay line
//     until its own threshold is ready.
// DEPTH=1 matches the single-register-stage convention already established
// for every other module in this design (see c2_addr_gen_tb.v).
//
// IMPLEMENTATION NOTE: DEPTH can run into the tens of thousands (x_delay's
// DEPTH scales with tk_sli*(IMG_WIDTH+1) at production SLI/IMG_WIDTH, e.g.
// ~12,825 at SLI=51/IMG_WIDTH=512). An earlier version of this module used a
// 2-D reg array (valid_sr[0:DEPTH-1]/data_sr[0:DEPTH-1]) with an explicit
// procedural `for` loop to shift and reset it -- functionally fine, but
// Quartus statically unrolls procedural for-loops during elaboration, and
// that hit its default 5000-iteration cap (error 10106) at production DEPTH,
// even though the loop always terminates. The `(* altera_loop_limit = N *)`
// attribute meant to raise that cap was not recognized by this Quartus Prime
// Lite 21.1 toolchain (Warning 10335), so instead of chasing pragma syntax,
// this version sidesteps the problem entirely: valid_sr/data_sr are flat
// bit-vectors shifted with a single concatenation each cycle, not an array
// walked by a loop. Zero procedural loop iterations at any DEPTH, and this
// is in fact the more standard Quartus idiom for shift-register (altshift_taps)
// inference. Verified to produce bit-identical timing to the old loop-based
// version by manual trace (see commit history / BUG_REPORT.md discussion).

module pipe_delay #(
    parameter integer DATA_WIDTH = 16,
    parameter integer DEPTH = 1
) (
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     in_valid,
    input  wire [DATA_WIDTH-1:0]    in_data,

    output wire                     out_valid,
    output wire [DATA_WIDTH-1:0]    out_data
);

    initial begin
        if (DEPTH < 1) $fatal(1, "pipe_delay: DEPTH must be >= 1");
    end

    reg [DEPTH-1:0]            valid_sr;
    reg [DATA_WIDTH*DEPTH-1:0] data_sr;

    generate
    if (DEPTH == 1) begin : D1
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                valid_sr <= 1'b0;
                data_sr  <= {DATA_WIDTH{1'b0}};
            end else begin
                valid_sr <= in_valid;
                data_sr  <= in_data;
            end
        end
    end else begin : DN
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                valid_sr <= {DEPTH{1'b0}};
                data_sr  <= {(DATA_WIDTH*DEPTH){1'b0}};
            end else begin
                // Newest sample enters at the LSB end; index DEPTH-1 (the
                // MSB end) is always the sample that arrived DEPTH cycles
                // ago -- identical shift semantics to the old valid_sr[k]
                // <= valid_sr[k-1] loop, just expressed as one vector op.
                valid_sr <= {valid_sr[DEPTH-2:0], in_valid};
                data_sr  <= {data_sr[DATA_WIDTH*(DEPTH-1)-1:0], in_data};
            end
        end
    end
    endgenerate

    assign out_valid = valid_sr[DEPTH-1];
    assign out_data  = data_sr[DATA_WIDTH*DEPTH-1 -: DATA_WIDTH];

endmodule
