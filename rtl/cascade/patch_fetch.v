// patch_fetch.v -- cuts the 32x32 CNN input window out of the pooled frame store.
//
//   window rows j-16 .. j+15, columns i-16 .. i+15 (pooled coordinates), indices clamped to the frame
//   (edge replicate). Streams 1024 bytes, row-major, one per clock, to cnn_core (in_valid/in_data).
//   Store read latency is 1 cycle; the address pipeline adds 1.
`timescale 1ns/1ps
module patch_fetch #(
    parameter integer WP = 400,          // pooled width
    parameter integer HP = 400,          // pooled height
    parameter integer AW = 18
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,          // pulse: begin fetching the window centred on (j,i)
    input  wire [9:0]    j,
    input  wire [9:0]    i,
    output reg  [AW-1:0] ps_raddr,
    input  wire [7:0]    ps_rdata,
    output reg           px_valid,
    output reg  [7:0]    px_data,
    output reg           busy
);
    localparam [1:0] S_IDLE = 2'd0, S_ROW = 2'd1, S_RUN = 2'd2, S_FLUSH = 2'd3;
    reg [1:0] st;
    reg signed [11:0] j0, i0;            // j-16, i-16
    reg [5:0] r, c;
    reg [AW-1:0] rowbase;
    reg [1:0] pv;                        // address -> data valid pipeline
    reg [2:0] flush;

    wire signed [11:0] jj_raw = j0 + $signed({6'd0, r});
    wire signed [11:0] ii_raw = i0 + $signed({6'd0, c});
    wire [11:0] jj = (jj_raw < 0) ? 12'd0 : (jj_raw > HP - 1) ? (HP - 1) : jj_raw[11:0];
    wire [11:0] ii = (ii_raw < 0) ? 12'd0 : (ii_raw > WP - 1) ? (WP - 1) : ii_raw[11:0];

    always @(posedge clk) begin
        if (!rstn) begin
            st <= S_IDLE; busy <= 1'b0; px_valid <= 1'b0; pv <= 2'b00; ps_raddr <= {AW{1'b0}};
            r <= 6'd0; c <= 6'd0; flush <= 3'd0;
        end else begin
            pv <= {pv[0], 1'b0};
            px_valid <= pv[1];
            px_data  <= ps_rdata;
            case (st)
                S_IDLE: begin
                    if (start) begin
                        j0 <= $signed({2'b00, j}) - 12'sd16;
                        i0 <= $signed({2'b00, i}) - 12'sd16;
                        r <= 6'd0; c <= 6'd0; busy <= 1'b1;
                        st <= S_ROW;
                    end
                end
                S_ROW: begin                     // row base address (one registered multiply per row)
                    rowbase <= jj * WP;
                    c <= 6'd0;
                    st <= S_RUN;
                end
                S_RUN: begin
                    ps_raddr <= rowbase + ii;
                    pv[0] <= 1'b1;
                    if (c == 6'd31) begin
                        if (r == 6'd31) begin
                            st <= S_FLUSH; flush <= 3'd4;
                        end else begin
                            r <= r + 1'b1; st <= S_ROW;
                        end
                    end else c <= c + 1'b1;
                end
                S_FLUSH: begin
                    if (flush == 0) begin busy <= 1'b0; st <= S_IDLE; end
                    else flush <= flush - 1'b1;
                end
            endcase
        end
    end
endmodule
