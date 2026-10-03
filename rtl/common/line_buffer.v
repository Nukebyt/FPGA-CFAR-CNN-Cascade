// line_buffer.v
// Parameterizable sliding window generator using a streaming line-delay +
// shift-register architecture (Quartus-friendly: RAM-inferrable row delays,
// no runtime-computed read addresses anywhere).
//
// REWRITE NOTE: the previous version of this module rebuilt the entire
// flattened SLI*SLI window every cycle by indexing into a 2-D register array
// (row_mem[rr][cc]) with runtime-computed rr/cc. Needing SLI^2 simultaneous
// arbitrary-address reads per cycle prevented Quartus from inferring row_mem
// as embedded (M9K) memory at all -- it synthesized as raw flip-flops plus a
// large mux/crossbar network instead, which is why Analysis & Synthesis
// reported 286,421 logic elements (2.5x the DE2-115's 114,480-LE budget) at
// only SLI=7 -- see README.md section 2 for the full writeup. This version
// replaces that with the standard FPGA image-processing pattern: SLI-1
// chained depth-IMG_WIDTH row-delay lines feeding SLI small, purely-static
// SLI-deep column shift registers (one per tap), giving window_pixels as a
// continuous readout of registers -- no computed addresses anywhere.
//
// SECOND FIX (BUG_LOG D21): this rewrite's row-delay lines were still not
// actually RAM-inferable in practice -- an inline `delay_mem[1:SLI-1]
// [0:IMG_WIDTH-1]` array, read/written through a shared blocking scratch
// variable inside one unrolled `for` loop, synthesized as raw flip-flops
// (0 Block Memory Bits per quartus_map's Resource Utilization by Entity
// report) instead of M10K, ballooning a single detector past the DE10-
// Standard's LAB budget. Fixed by moving the row-delay storage into
// row_delay_mem.v, one clean single-port-RAM module per stage, chained
// stage-to-stage via pure combinational wires (never through a blocking
// variable shared across always blocks) -- see that file's header.
//
// EVERYTHING IN ONE ALWAYS BLOCK (important, not a style choice): two
// earlier versions of this file split the row-delay chain, the per-tap
// column shift registers, and/or the row-delay stages themselves across
// SEPARATE module instances / always blocks, each connected by a plain
// wire. Both attempts produced real, silent bugs: reading a value that was
// just computed by a DIFFERENT always block (even at the "same" posedge,
// even within the same module) sees that register's PRE-edge value, not
// the value it computes THIS edge -- the cross-module/cross-block NBA lag
// this project's own Latency Ledger documents for module-to-module
// connections (BUG_REPORT.md), which turns out to apply just as much
// between two always blocks in the SAME file. The first attempt (chaining
// SLI-1 separate line_delay module instances) let this lag compound once
// per stage, producing a diagonal shear across the window (row r's content
// one column further stale than row r+1's). The second attempt fixed the
// row-delay chain but still read its output (taps[k]) from a SEPARATE
// col_shift always block, producing a uniform one-column-stale skew on
// every row except the bottom one (tap 0, fed directly from the
// testbench/port-level pixel_in, which has no such lag). Neither was
// obviously wrong from the output shape -- both were caught only by
// comparing individual windows against tb/expected_sums.txt, not by
// inspection. The fix, applied here: the ENTIRE per-cycle pipeline (row
// delay chain, its validity tracking, the column shift registers, and the
// warm-up settle chain) lives in ONE always block, with every value that
// flows from one conceptual stage to the next inside this cycle carried by
// a BLOCKING scratch variable (stage_in/stage_out/stage_valid_in), so nothing
// ever crosses an always-block boundary before being used. Only the actual
// per-cycle STATE (col_shift, delay_mem, warmup_done, col_cnt, ...) is
// committed via nonblocking assignment, exactly once, at the natural end of
// the block.
//
// VALIDITY TIMING -- a real, deliberate difference from the old module, not
// a bug: the old design's row_mem was written by a direct index every
// cycle, so a position became valid the instant it was first written, with
// no settling time. A real delay-chain-based design cannot do that: data
// takes real, accumulating cycles to propagate through SLI-1 chained delay
// stages plus each one's own SLI-deep shift register. window_valid only
// asserts once every tap's col_shift row is provably populated with real,
// streamed data -- tracked structurally via a valid flag chained through
// the SAME hardware the data itself flows through (so it inherits the
// exact real latency by construction, not a separately hand-derived formula
// -- attempted first, and confirmed wrong: the true startup shortfall
// depends on IMG_WIDTH as well as SLI, not SLI alone). This costs a few
// extra cycles of startup latency once per frame compared to the old
// (not physically realizable at large SLI) instant-access assumption, in
// exchange for zero risk of reading undefined data -- one of the two
// boundary-handling options WeibullCFAR_Floating.m's own header already
// flags as legitimate for the streaming architecture ("simply not declare
// detections in the first/last tsli rows/cols of a frame").

module line_buffer #(
    parameter integer SLI = 7,               // window size (odd)
    parameter integer IMG_WIDTH = 640,       // image columns
    parameter integer IMG_HEIGHT = 480,      // image rows
    parameter integer DATA_WIDTH = 16        // pixel data width
) (
    input  wire                         clk,
    input  wire                         rstn,
    input  wire                         pixel_in_valid,
    input  wire signed [DATA_WIDTH-1:0] pixel_in,

    output reg                          window_valid,
    output wire signed [SLI*SLI*DATA_WIDTH-1:0]  window_pixels /* flattened [SLI*SLI] */
);

    // Sanity bound only (catches typos / runaway parameters) -- NOT an
    // architectural limit the way the old MAX_SLI=21 was. This design's
    // resource cost scales O(SLI) in memory and O(SLI^2) only in flip-flops
    // (cheap), so there is no structural reason to cap SLI at 21.
    localparam integer MAX_SLI_SANITY = 199;
    initial begin
        if (SLI < 1 || SLI > MAX_SLI_SANITY) begin
            $fatal(1, "SLI out of sane range (1..%0d)", MAX_SLI_SANITY);
        end
    end

    // col_shift[k][0] = tap k's value THIS cycle (freshest / just arrived).
    // col_shift[k][SLI-1] = tap k's value from SLI-1 valid-cycles ago
    // (oldest surviving column sample for that row-offset). tap k = the
    // pixel that arrived exactly k rows ago at this same column (k=0 is
    // the current row, no delay). Both the row-delay chain that produces
    // each tap and the shift that stores it happen in ONE always block
    // below (see file header) -- no addressing, no cross-block reads.
    reg signed [DATA_WIDTH-1:0] col_shift [0:SLI-1][0:SLI-1];

    reg warmup_done;
    integer col_cnt;

    generate
    if (SLI == 1) begin : TRIVIAL
        // No row/column history needed at all -- the "window" is just the
        // current pixel.
        integer m0;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                col_shift[0][0] <= {DATA_WIDTH{1'b0}};
                warmup_done <= 1'b0;
                col_cnt <= 0;
                window_valid <= 1'b0;
            end else if (pixel_in_valid) begin
                col_shift[0][0] <= pixel_in;
                warmup_done <= 1'b1;
                window_valid <= 1'b1;
                col_cnt <= 0;
            end else begin
                window_valid <= 1'b0;
            end
        end
    end else begin : MAIN

        // Row-delay storage: SLI-1 chained single-port RAMs (row_delay_mem),
        // NOT an inline 2-D reg array. See row_delay_mem.v's header for the
        // full history -- short version: an inline 2-D array read/written
        // through a shared blocking scratch variable in one loop didn't
        // infer as RAM at all (0 Block Memory Bits, ~94,000 phantom
        // registers across 3 instances, blowing a detector's whole LAB
        // budget); splitting it into separate per-stage modules with an
        // ASYNC read (`assign dout = mem[addr]`, matching how this file's
        // ROM neighbors are coded) ALSO didn't infer, because Quartus's
        // default single-port RAM inference requires a SYNCHRONOUS read,
        // full stop -- confirmed via quartus_map's own diagnostic (Info
        // 276007, "uninferred due to asynchronous read logic"), not
        // guessed. row_delay_mem.v now registers its read.
        //
        // That registered read adds exactly one clock of latency PER
        // STAGE relative to the old (broken) async version. Compensated by
        // giving every stage DEPTH=IMG_WIDTH-1 instead of IMG_WIDTH
        // (uniformly -- unlike an earlier, now-obsolete attempt at this fix
        // that special-cased stage 1, back when only an EXTERNAL link
        // register between stages needed compensating; now every stage's
        // own internal register needs it, stage 1 included) so tap kk's
        // total delay stays exactly kk*IMG_WIDTH cycles: sum of kk stage
        // depths (kk*(IMG_WIDTH-1)) + kk internal registers passed through
        // = kk*IMG_WIDTH. Getting this wrong would reintroduce the
        // diagonal-shear-by-column-offset bug class this file's header
        // already documents from an earlier rewrite -- verified bit-exact
        // against front_end3_tb/gengamma_top_tb/gap tests, not derived on
        // paper alone.
        //
        // rdm_dout is now a REGISTER's output (row_delay_mem's own), not an
        // async memory read, so chaining it directly into the next stage's
        // din (plain wires, `rdm_din[grd]=rdm_dout[grd-1]`) is the standard,
        // unremarkable register-to-RAM-input pattern -- no separate link
        // register needed for the data path.
        //
        // FILLCNT THRESHOLD IS NOT STAGE_DEPTH -- a real bug, caught the
        // same way (front_end3_tb's bit-exact per-window check, uniform
        // one-column shift signature) as the very first attempt at this
        // compensation. fillcnt/stage_valid_in is a purely LOGICAL count of
        // "how many cycles until this conceptual stage's output is real,"
        // which has ALWAYS been exactly IMG_WIDTH (independent of how the
        // memory happens to be implemented) -- conflating it with
        // STAGE_DEPTH (an implementation detail of the addressing, which
        // legitimately did shrink by 1) made validity assert one cycle
        // before the correspondingly-delayed real data actually arrived,
        // for the SAME underlying reason as before, just relocated: fillcnt
        // must compare against IMG_WIDTH, while ONLY the address wraparound
        // uses the reduced STAGE_DEPTH. Once that split is made correctly,
        // stage kk's OWN validity and stage kk's OWN dout become real on
        // the exact same cycle again (verified by hand-trace against a
        // small IMG_WIDTH example) -- so the plain same-cycle blocking
        // chain for stage_valid_in (no extra register at all) is correct,
        // exactly as it was before ANY of this RAM-inference work started.
        localparam integer STAGE_DEPTH = IMG_WIDTH - 1;

        wire signed [DATA_WIDTH-1:0] rdm_din  [1:SLI-1];
        wire signed [DATA_WIDTH-1:0] rdm_dout [1:SLI-1];
        reg  [31:0] delay_addr [1:SLI-1];
        reg  [31:0] fillcnt    [1:SLI-1];
        reg         settle_delay [0:SLI-2];

        assign rdm_din[1] = pixel_in;
        genvar grd;
        for (grd = 2; grd < SLI; grd = grd + 1) begin : RDM_CHAIN
            assign rdm_din[grd] = rdm_dout[grd-1];
        end
        for (grd = 1; grd < SLI; grd = grd + 1) begin : RDM_INST
            row_delay_mem #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(STAGE_DEPTH)) u_rdm (
                .clk  (clk),
                .we   (pixel_in_valid),
                .addr (delay_addr[grd]),
                .din  (rdm_din[grd]),
                .dout (rdm_dout[grd])
            );
        end

        integer kk, m, s;
        reg                         stage_valid_in;
        reg [31:0]                  next_fillcnt;

        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                for (kk = 1; kk < SLI; kk = kk + 1) begin
                    delay_addr[kk] <= 0;
                    fillcnt[kk]    <= 0;
                end
                for (kk = 0; kk < SLI; kk = kk + 1)
                    for (m = 0; m < SLI; m = m + 1)
                        col_shift[kk][m] <= {DATA_WIDTH{1'b0}};
                for (s = 0; s < SLI-1; s = s + 1) settle_delay[s] <= 1'b0;
                warmup_done  <= 1'b0;
                col_cnt      <= 0;
                window_valid <= 1'b0;
            end else if (pixel_in_valid) begin
                // ---- Stage: tap 0 (current row, zero delay) -------------
                for (m = SLI-1; m > 0; m = m - 1)
                    col_shift[0][m] <= col_shift[0][m-1];
                col_shift[0][0] <= pixel_in;

                // ---- Stage: row-delay chain + column shift, taps 1..SLI-1
                // The row delay itself (rdm_dout[kk], an IMG_WIDTH-cycle-old
                // sample) is now supplied structurally by the row_delay_mem
                // chain above -- this loop only threads stage_valid_in
                // (BLOCKING, so validity ripples through all SLI-1 stages
                // within this single clock edge, same as before) and does
                // the column-shift bookkeeping.
                stage_valid_in = 1'b1;
                for (kk = 1; kk < SLI; kk = kk + 1) begin
                    delay_addr[kk] <= (delay_addr[kk] == STAGE_DEPTH-1) ? 0 : delay_addr[kk] + 1;

                    // Threshold is IMG_WIDTH (the logical per-stage
                    // latency), NOT STAGE_DEPTH (an addressing-only
                    // implementation detail) -- see header.
                    if (stage_valid_in)
                        next_fillcnt = (fillcnt[kk] < IMG_WIDTH+1) ? fillcnt[kk] + 1 : fillcnt[kk];
                    else
                        next_fillcnt = 0;
                    fillcnt[kk] <= next_fillcnt;

                    for (m = SLI-1; m > 0; m = m - 1)
                        col_shift[kk][m] <= col_shift[kk][m-1];
                    col_shift[kk][0] <= rdm_dout[kk];

                    // Register this stage's own validity for the NEXT
                    // stage to consume one cycle from now (see above).
                    // No-op for the last stage (kk == SLI-1), no successor.
                    stage_valid_in = (next_fillcnt > IMG_WIDTH);
                end
                // stage_valid_in now holds tap_valid[SLI-1] (this cycle),
                // i.e. whether the slowest tap is carrying real data.

                // ---- Global warm-up settle (see file header) -------------
                // col_shift[SLI-1][SLI-1] (the single slowest-filling
                // element in the whole window) needs SLI-1 MORE cycles
                // after tap SLI-1 itself becomes real, for that element to
                // reach the far end of ITS OWN shift register. Plain shift
                // register, not a threshold -- no off-by-one possible.
                settle_delay[0] <= stage_valid_in;
                for (s = 1; s < SLI-1; s = s + 1) settle_delay[s] <= settle_delay[s-1];
                warmup_done <= warmup_done | settle_delay[SLI-2];

                // ---- Per-row column-boundary validity (BUG_REPORT.md #4) -
                window_valid <= warmup_done && (col_cnt >= SLI-1);
                if (col_cnt == IMG_WIDTH-1) col_cnt <= 0;
                else col_cnt <= col_cnt + 1;
            end else begin
                for (kk = 1; kk < SLI; kk = kk + 1) fillcnt[kk] <= 0;
                for (s = 0; s < SLI-1; s = s + 1) settle_delay[s] <= 1'b0;
                window_valid <= 1'b0;
            end
        end
    end
    endgenerate

    // ---- Flatten to window_pixels -------------------------------------------
    // window ROW r (0=top .. SLI-1=bottom) = tap (SLI-1-r) [tap 0 = current/
    // bottom row, tap SLI-1 = oldest/top row]. window COLUMN c (0=left ..
    // SLI-1=right) = col_shift position (SLI-1-c) [position 0 = newest/
    // rightmost, position SLI-1 = oldest/leftmost]. Packed MSB-first
    // top-left .. bottom-right, exactly matching the bit layout
    // window_sum.v already expects (unchanged from the original module).
    // All indices below are elaboration-time constants (genvar-derived
    // localparams), so this is pure static wiring -- zero-cost routing,
    // not logic.
    genvar gr, gc;
    generate
        for (gr = 0; gr < SLI; gr = gr + 1) begin : WR
            for (gc = 0; gc < SLI; gc = gc + 1) begin : WC
                localparam integer IDX    = gr*SLI + gc;
                localparam integer BITPOS = (SLI*SLI - 1 - IDX) * DATA_WIDTH;
                assign window_pixels[BITPOS +: DATA_WIDTH] = col_shift[SLI-1-gr][SLI-1-gc];
            end
        end
    endgenerate

endmodule
