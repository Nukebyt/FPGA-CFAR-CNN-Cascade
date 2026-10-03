// row_delay_mem.v
// One row-delay stage's storage, split out of line_buffer.v into its own
// module purely so Quartus can infer it as a single-port RAM (M10K).
//
// WHY THIS EXISTS: line_buffer.v's row-delay chain used to be a single
// `reg [...] delay_mem [1:SLI-1][0:IMG_WIDTH-1]` array, read/written by an
// unrolled `for` loop inside line_buffer's one big always block, with the
// read value threaded through a blocking scratch variable into the NEXT
// stage's write and into that stage's own col_shift/fillcnt logic, all in
// the same clock edge. That is a completely valid, deliberate design (see
// line_buffer.v's header for why it MUST all live in one always block) --
// but interleaving reads/writes of several different array instances
// through a shared blocking variable, in a loop, defeated Quartus's RAM
// inference silently: quartus_map's Resource Utilization by Entity report
// showed 0 Block Memory Bits for these arrays, meaning each one synthesized
// as ~IMG_WIDTH*DATA_WIDTH flip-flops instead of one small M10K -- at
// SLI=17, IMG_WIDTH=64 that is roughly 94,000 registers across the three
// line_buffer instances of a single detector, which alone blew a GenGamma
// top-level past the DE10-Standard's 5CSXFC6D6 LAB budget (6530 LABs
// needed vs 4191 available) despite the *real* logic being modest.
//
// Fix: give each stage its OWN, textbook single-port-RAM module -- one
// array, one always block, nothing else touching it.
//
// REGISTERED READ, NOT ASYNC -- this took three attempts to get right.
// The first version here used `assign dout = mem[addr]` (async/
// combinational read, matching how log_amp_rom.v/gengamma_num_rom.v are
// coded), reasoning that a chain of these looked exactly like a proven
// pattern already used elsewhere in this design. Two follow-up theories
// (that the failure was Quartus declining to infer RAM when one memory's
// output feeds directly into another memory's write-data port; and that a
// `ramstyle` attribute could force it) were both wrong and both verified
// wrong empirically (a scratch `_quartus/_ram_probe/` project, small enough
// to synthesize in ~30s instead of ~12min, iterated this fast). The ACTUAL
// answer was printed the whole time, just never grepped for:
//   Info (276007): RAM logic "...|mem" is uninferred due to asynchronous
//   read logic
// Quartus's default single-port RAM inference requires a SYNCHRONOUS read
// (dout registered here, not `assign`ed) -- full stop, independent of any
// fanout pattern. ROM inference (log_amp_rom.v etc.) evidently tolerates
// unregistered output via a different recognition path; generic RAM
// inference does not. Confirmed on the same scratch probe: switching this
// module's read from `assign` to a registered `always` block took it from
// 4246 logic cells / 0 RAM segments to 39 logic cells / 48 RAM segments for
// an identical 3-stage chain.
//
// The registered read adds exactly one clock of latency PER INSTANCE
// relative to the (broken) async version. line_buffer.v compensates by
// giving every chained stage a DEPTH one less than before (IMG_WIDTH-1
// instead of IMG_WIDTH, uniformly -- see its header), and by delaying the
// stage-to-stage VALIDITY handshake by that same one cycle
// (`valid_link[]`), so total per-tap latency is unchanged.
module row_delay_mem #(
    parameter integer DATA_WIDTH = 16,
    parameter integer DEPTH      = 640
) (
    input  wire                         clk,
    input  wire                         we,
    input  wire [31:0]                  addr,
    input  wire signed [DATA_WIDTH-1:0] din,
    output reg  signed [DATA_WIDTH-1:0] dout
);
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[addr] <= din;
        dout <= mem[addr];
    end
endmodule
