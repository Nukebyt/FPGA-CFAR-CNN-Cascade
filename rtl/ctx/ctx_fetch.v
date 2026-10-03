// ctx_fetch.v -- cuts the 32x32 CONTEXT window (input of the context tower) out of the pooled frame store.
//   P4[r][c] = (P[2r][2c] + P[2r][2c+1] + P[2r+1][2c] + P[2r+1][2c+1] + 2) >> 2   (a virtual 200x200 store: 2x2 pool of the pooled frame)
//   window rows (j>>1)-16 .. (j>>1)+15, columns (i>>1)-16 .. (i>>1)+15, indices clamped to [0, HP/2-1] / [0, WP/2-1] (edge replicate),
//   streamed row-major, one P4 pixel per 4 store reads (4096 reads per window), to cnn_core_ctx (in_kind = 1).
// Golden model: _comparison/fixedpoint/extract_fx_events.py (pool2 + edge pad).  Store read latency as patch_fetch.v.
`timescale 1ns/1ps
module ctx_fetch #(
    parameter integer WP = 400, parameter integer HP = 400, parameter integer AW = 18
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,
    input  wire [9:0]    j,
    input  wire [9:0]    i,
    output reg  [AW-1:0] ps_raddr,
    input  wire [7:0]    ps_rdata,
    output reg           px_valid,
    output reg  [7:0]    px_data,
    output reg           busy
);
    localparam integer HQ = HP / 2, WQ = WP / 2;
    localparam [1:0] S_IDLE = 2'd0, S_ROW = 2'd1, S_RUN = 2'd2, S_FLUSH = 2'd3;
    reg [1:0] st;
    reg signed [11:0] j0, i0;            // (j>>1)-16, (i>>1)-16
    reg [5:0] r, c;
    reg [1:0] q;
    reg [AW-1:0] rowbase;
    reg [2:0] flush;
    reg [2:0] pv; reg [1:0] pq0, pq1;     // address -> data pipeline with the sub-index tags
    reg [9:0] acc;

    wire signed [11:0] rr_raw = j0 + $signed({6'd0, r});
    wire signed [11:0] cc_raw = i0 + $signed({6'd0, c});
    wire [11:0] rr = (rr_raw < 0) ? 12'd0 : (rr_raw > HQ - 1) ? (HQ - 1) : rr_raw[11:0];
    wire [11:0] cc = (cc_raw < 0) ? 12'd0 : (cc_raw > WQ - 1) ? (WQ - 1) : cc_raw[11:0];

    always @(posedge clk) begin
        if (!rstn) begin
            st <= S_IDLE; busy <= 1'b0; px_valid <= 1'b0; pv <= 3'b000; ps_raddr <= {AW{1'b0}};
            r <= 6'd0; c <= 6'd0; q <= 2'd0; flush <= 3'd0; acc <= 10'd0;
        end else begin
            pv <= {pv[1:0], 1'b0}; pq1 <= pq0;
            px_valid <= 1'b0;
            // data arriving this cycle (issued two cycles ago): accumulate the 2x2 block
            if (pv[1]) begin
                if (pq1 == 2'd0) acc <= {2'b00, ps_rdata};
                else if (pq1 != 2'd3) acc <= acc + ps_rdata;
                else begin px_data <= (acc + ps_rdata + 10'd2) >> 2; px_valid <= 1'b1; end
            end
            case (st)
                S_IDLE: begin
                    if (start) begin
                        j0 <= $signed({2'b00, j >> 1}) - 12'sd16;
                        i0 <= $signed({2'b00, i >> 1}) - 12'sd16;
                        r <= 6'd0; c <= 6'd0; q <= 2'd0; busy <= 1'b1; st <= S_ROW;
                    end
                end
                S_ROW: begin rowbase <= (rr << 1) * WP; c <= 6'd0; q <= 2'd0; st <= S_RUN; end
                S_RUN: begin
                    ps_raddr <= rowbase + (q[1] ? WP : 0) + (cc << 1) + q[0];
                    pv[0] <= 1'b1; pq0 <= q;
                    if (q != 2'd3) q <= q + 1'b1;
                    else begin
                        q <= 2'd0;
                        if (c == 6'd31) begin
                            if (r == 6'd31) begin st <= S_FLUSH; flush <= 3'd5; end
                            else begin r <= r + 1'b1; st <= S_ROW; end
                        end else c <= c + 1'b1;
                    end
                end
                S_FLUSH: begin
                    if (flush == 0) begin busy <= 1'b0; st <= S_IDLE; end
                    else flush <= flush - 1'b1;
                end
            endcase
        end
    end
endmodule
