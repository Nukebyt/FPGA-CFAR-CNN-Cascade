// box_sum_inc.v
// ---------------------------------------------------------------------------
// Incremental separable box-sum (window total, and its concentric guard hole)
// over a SINGLE plane -- no squaring, no cubing, just adds/subtracts. The
// plane's VALUE (x, x^2, or x^3) is computed exactly once, upstream, by
// power_expand; this module only sums whatever it is given.
//
// This is window_sum.v's proven incremental row-maintenance technique
// (O(SLI) work per cycle via evict/insert, not O(SLI^2) recompute), stripped
// of the squaring it used to do internally -- see rtl/common/power_expand.v
// and its header for why the squaring moved upstream (it lets each power be
// computed ONCE per pixel instead of twice, halving the multiplier count).
// Because there is no multiply anywhere in this module, three instances (one
// per moment) cost ALMs only, never DSP blocks.
// ---------------------------------------------------------------------------

module box_sum_inc #(
    parameter integer SLI = 17,
    parameter integer GUARD = 13,
    parameter integer DATA_WIDTH = 16,
    parameter integer SUM_WIDTH  = 23,
    parameter integer SIGNED_DATA = 1   // 1: sign-extend on widen; 0: zero-extend
) (
    input  wire                                   clk,
    input  wire                                   rstn,
    input  wire                                   window_valid,
    input  wire [SLI*SLI*DATA_WIDTH-1:0]          window_plane, // MSB-first top-left..bottom-right

    output reg                                    sums_valid,
    output reg signed [SUM_WIDTH-1:0]             sum_window,
    output reg signed [SUM_WIDTH-1:0]             sum_guard
);

    localparam integer G_START = (SLI - GUARD) / 2;
    localparam integer G_END   = G_START + GUARD - 1;

    initial begin
        if (GUARD >= SLI) $fatal(1, "GUARD must be smaller than SLI");
    end

    function automatic [DATA_WIDTH-1:0] wp_at;
        input integer row;
        input integer col;
        integer idx, bitpos;
        begin
            idx = row*SLI + col;
            bitpos = (SLI*SLI - 1 - idx) * DATA_WIDTH;
            wp_at = window_plane[bitpos +: DATA_WIDTH];
        end
    endfunction

    function automatic signed [SUM_WIDTH-1:0] widen;
        input [DATA_WIDTH-1:0] v;
        begin
            if (SIGNED_DATA)
                widen = {{(SUM_WIDTH-DATA_WIDTH){v[DATA_WIDTH-1]}}, v};
            else
                widen = {{(SUM_WIDTH-DATA_WIDTH){1'b0}}, v};
        end
    endfunction

    reg signed [SUM_WIDTH-1:0] row_sum  [0:SLI-1];
    reg signed [SUM_WIDTH-1:0] grow_sum [0:SLI-1];
    reg [DATA_WIDTH-1:0] prev_col0 [0:SLI-1];
    reg [DATA_WIDTH-1:0] prev_colg [0:SLI-1];

    reg seen_first;
    integer r, c;
    reg signed [SUM_WIDTH-1:0] new_row_sum, new_grow_sum;
    reg signed [SUM_WIDTH-1:0] acc, gacc;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sums_valid <= 1'b0;
            sum_window <= 0; sum_guard <= 0;
            seen_first <= 1'b0;
            for (r = 0; r < SLI; r = r + 1) begin
                row_sum[r] <= 0; grow_sum[r] <= 0;
                prev_col0[r] <= 0; prev_colg[r] <= 0;
            end
        end else if (window_valid) begin
            acc = 0; gacc = 0;
            if (!seen_first) begin
                for (r = 0; r < SLI; r = r + 1) begin
                    new_row_sum = 0; new_grow_sum = 0;
                    for (c = 0; c < SLI; c = c + 1) begin
                        new_row_sum = new_row_sum + widen(wp_at(r,c));
                        if (c >= G_START && c <= G_END)
                            new_grow_sum = new_grow_sum + widen(wp_at(r,c));
                    end
                    row_sum[r]  <= new_row_sum;
                    grow_sum[r] <= new_grow_sum;
                    prev_col0[r] <= wp_at(r, 0);
                    prev_colg[r] <= wp_at(r, G_START);
                    acc  = acc  + new_row_sum;
                    if (r >= G_START && r <= G_END) gacc = gacc + new_grow_sum;
                end
                seen_first <= 1'b1;
            end else begin
                for (r = 0; r < SLI; r = r + 1) begin
                    new_row_sum = row_sum[r] + widen(wp_at(r,SLI-1)) - widen(prev_col0[r]);
                    row_sum[r] <= new_row_sum;
                    if (r >= G_START && r <= G_END) begin
                        new_grow_sum = grow_sum[r] + widen(wp_at(r,G_END)) - widen(prev_colg[r]);
                        grow_sum[r] <= new_grow_sum;
                    end else begin
                        new_grow_sum = grow_sum[r];
                    end
                    prev_col0[r] <= wp_at(r, 0);
                    prev_colg[r] <= wp_at(r, G_START);
                    acc  = acc  + new_row_sum;
                    if (r >= G_START && r <= G_END) gacc = gacc + new_grow_sum;
                end
            end
            sum_window <= acc;
            sum_guard  <= gacc;
            sums_valid <= 1'b1;
        end else begin
            sums_valid <= 1'b0;
            seen_first <= 1'b0;
        end
    end

endmodule
