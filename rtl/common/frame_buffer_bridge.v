// frame_buffer_bridge.v
// Fixes KNOWN_ISSUE_GAP_INTOLERANCE.md via that document's own recommended
// Option 1: "Buffer at the bridge, not the core." front_end3/line_buffer.v
// need a genuinely continuous, gap-free pixel_in_valid stream to ever reach
// warmup_done -- real HPS/Avalon-MM traffic (tens to 100+ cycles between
// MMIO register writes) can never provide that directly. This module sits
// between a gapped host write interface and a detector core's pixel_in
// port: the host fills an on-chip frame buffer at its own slow, gapped
// pace; once a FULL frame has been written, this module's own FSM drains it
// into the core continuously, one pixel per clock, no gaps -- satisfying
// line_buffer's hard requirement without touching front_end3/line_buffer.v
// at all (both remain exactly as already verified by the full regression
// suite -- front_end3_tb, gengamma_top_tb, the gap/stall tests, all 5
// detector top-level tests -- none of which need re-running because of
// this module).
//
// ALSO HANDLES THE OTHER HALF OF THE SAME PROBLEM CLASS: line_buffer.v has
// no frame-boundary concept at all (BUG_LOG D20) -- consecutive frames are
// NOT independent without an explicit core reset between them. This module
// pulses core_rstn_out low for RESET_HOLD cycles before every drain starts,
// the exact same per-frame-reset pattern gengamma_de10_top.v's ST_RESET
// state already uses and has verified (gengamma_de10_top_tb.v: correct,
// repeatable detection across 3 independent frame loops).
//
// SINGLE-BUFFERED, NOT PING-PONG -- a deliberate scope choice, not an
// oversight: the host must wait for drain_done before starting the next
// frame_start (asserting wr_en during DRAIN is ignored, see FSM below). A
// double-buffered version would let the host start filling the NEXT frame
// while THIS one drains, improving throughput -- left as a documented future
// extension (see "Not yet done" below), not required to fix the actual
// zero-detections failure mode this module exists to solve.
//
// SINGLE-PORT RAM, matching row_delay_mem.v's own proven synchronous-read
// inference pattern -- safe here because WRITE (host fill) and READ (core
// drain) phases are temporally disjoint by FSM construction (one frame is
// never being written and drained at the same time), so one address bus
// shared between the two phases is sufficient; no dual-port needed.
module frame_buffer_bridge #(
    parameter integer IMG_WIDTH  = 512,
    parameter integer IMG_HEIGHT = 512,
    parameter integer DATA_WIDTH = 8,
    parameter integer RESET_HOLD = 8    // cycles to hold core_rstn_out low before each drain -- matches gengamma_de10_top.v's own RESET_HOLD default
) (
    input  wire                     clk,
    input  wire                     rstn,

    // ---- Host write side (gapped, tolerant of any idle pattern) ----------
    input  wire                     frame_start,  // pulse: host is about to write a new frame from pixel 0
    input  wire                     wr_en,        // pulse per pixel; ignored outside ST_FILL
    input  wire [DATA_WIDTH-1:0]    wr_data,
    output wire                     ready_for_frame, // high when a new frame_start may be accepted (idle, not mid-fill/mid-drain)
    output wire [31:0]              wr_count,     // pixels received so far this frame, for host-side progress/debug

    // ---- Detector core drive side (gap-free, exactly front_end3's contract)
    output wire                     core_rstn_out,
    output wire                     pixel_out_valid,
    output wire [DATA_WIDTH-1:0]    pixel_out,

    // ---- Status -------------------------------------------------------------
    output wire                     draining,     // high during the gap-free drain (for an LED, etc.)
    output wire                     drain_done    // one-cycle pulse when a full frame has been fed to the core
);
    localparam integer N_PIX = IMG_WIDTH * IMG_HEIGHT;
    localparam integer ADDR_W = $clog2(N_PIX);

    // ---- Frame storage: single-port RAM, synchronous read (row_delay_mem.v's
    // proven inference pattern -- registered dout, not `assign`, see that
    // file's header for exactly why async read defeats Quartus's inference).
    reg [DATA_WIDTH-1:0] mem [0:N_PIX-1];
    reg [ADDR_W-1:0]     mem_addr;
    reg                  mem_we;
    reg [DATA_WIDTH-1:0] mem_din;
    reg [DATA_WIDTH-1:0] mem_dout;
    always @(posedge clk) begin
        if (mem_we) mem[mem_addr] <= mem_din;
        mem_dout <= mem[mem_addr];
    end

    localparam ST_IDLE = 3'd0, ST_FILL = 3'd1, ST_RESET = 3'd2, ST_DRAIN = 3'd3, ST_DRAIN_WAIT = 3'd4, ST_DONE = 3'd5;
    reg [2:0]        state;
    reg [ADDR_W-1:0]  fill_addr;
    reg [ADDR_W-1:0]  drain_addr;
    reg [3:0]         reset_cnt;
    reg               core_rstn_r;
    reg               pixel_out_valid_r;
    reg               draining_r;
    reg               drain_done_r;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state <= ST_IDLE;
            fill_addr <= 0; drain_addr <= 0; reset_cnt <= 0;
            core_rstn_r <= 1'b0; pixel_out_valid_r <= 1'b0;
            draining_r <= 1'b0; drain_done_r <= 1'b0;
            mem_we <= 1'b0; mem_addr <= 0; mem_din <= 0;
        end else begin
            drain_done_r <= 1'b0;
            mem_we <= 1'b0;

            case (state)
                ST_IDLE: begin
                    core_rstn_r <= 1'b1;  // core idle between frames; no pixel activity, rstn held high (core's own warmup state is irrelevant until the next drain starts, which itself re-pulses rstn below)
                    pixel_out_valid_r <= 1'b0;
                    if (frame_start) begin
                        fill_addr <= 0;
                        state     <= ST_FILL;
                    end
                end
                ST_FILL: begin
                    // Accept writes at WHATEVER pace the host manages --
                    // this is the entire point of this module. wr_en can be
                    // gapped arbitrarily; only the cycles where it's actually
                    // asserted advance fill_addr.
                    if (wr_en && fill_addr < N_PIX) begin
                        mem_we    <= 1'b1;
                        mem_addr  <= fill_addr;
                        mem_din   <= wr_data;
                        fill_addr <= fill_addr + 1'b1;
                        if (fill_addr == N_PIX-1) begin
                            reset_cnt <= 0;
                            state     <= ST_RESET;
                        end
                    end
                end
                ST_RESET: begin
                    // Per-frame core reset (D20's requirement, same pattern
                    // gengamma_de10_top.v's ST_RESET already uses/verified).
                    core_rstn_r <= 1'b0;
                    if (reset_cnt < RESET_HOLD) begin
                        reset_cnt <= reset_cnt + 4'd1;
                    end else begin
                        core_rstn_r <= 1'b1;
                        drain_addr  <= 0;
                        mem_addr    <= 0;    // prime the read pipeline one cycle early (registered read, see above)
                        state       <= ST_DRAIN;
                    end
                end
                ST_DRAIN: begin
                    // Gap-free, one real pixel every single cycle -- the
                    // actual fix: front_end3 never sees the gaps the host
                    // write side had.
                    draining_r <= 1'b1;
                    mem_addr   <= (drain_addr == N_PIX-1) ? drain_addr : drain_addr + 1'b1;
                    pixel_out_valid_r <= 1'b1;
                    if (drain_addr == N_PIX-1) begin
                        state <= ST_DRAIN_WAIT;
                    end else begin
                        drain_addr <= drain_addr + 1'b1;
                    end
                end
                ST_DRAIN_WAIT: begin
                    // mem_dout is one cycle behind mem_addr (registered
                    // read) -- one more cycle to present the LAST pixel
                    // before dropping valid.
                    pixel_out_valid_r <= 1'b0;
                    draining_r        <= 1'b0;
                    drain_done_r      <= 1'b1;
                    state             <= ST_IDLE;
                end
                default: state <= ST_IDLE;
            endcase
        end
    end

    assign core_rstn_out   = core_rstn_r;
    assign pixel_out_valid = pixel_out_valid_r;
    assign pixel_out       = mem_dout;
    assign draining        = draining_r;
    assign drain_done      = drain_done_r;
    assign ready_for_frame = (state == ST_IDLE);
    assign wr_count        = {{(32-ADDR_W){1'b0}}, fill_addr};

endmodule

// NOT YET DONE (documented, not silently skipped):
//   - Ping-pong double-buffering, to let the host start filling the next
//     frame while this one drains (throughput improvement, not a
//     correctness requirement -- see file header).
//   - A concrete Avalon-MM slave wrapper around this module's wr_en/wr_data/
//     frame_start ports (this module is transport-agnostic by design; the
//     actual HPS-facing register map is a separate, not-yet-built piece).
//   - Synthesis/resource-probe run (BRAM cost at production IMG_WIDTH=
//     IMG_HEIGHT=512: 512*512*8 = 2,097,152 bits, ~37% of this chip's
//     5,662,720 total block-memory bits on top of whichever single
//     detector core it feeds -- projected to fit comfortably alongside any
//     of the 5 already-measured cores' own block-memory use, but not yet
//     confirmed with a real quartus_map run).
