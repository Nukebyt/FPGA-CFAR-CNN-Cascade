// pool_store_writer.v -- builds the "pooled frame store" the CNN stage reads from.
//
//   pixel (8-bit) --QROM--> q8 --2x2 average, round-half-up--> P[j][i] = (a+b+c+d+2) >> 2
//
// The frame grid is fixed (pairs of rows 2j,2j+1 and columns 2i,2i+1); the CNN patch is later cut out of
// this pooled image, so no full-resolution frame is ever stored (1/4 of the memory).
//
// Raster input. Even rows: the pair {q[2i], q[2i+1]} is saved in a W/2 x 16-bit row buffer.
// Odd rows: when the odd column arrives, the saved pair plus the current pair give P[j][i]; it is written
// to the store at the next sequential address (pooled raster order). Pixel gaps are tolerated.
// `frame_in_done` is high in the same cycle as the last pooled write is presented (ps_we/ps_addr/ps_wdata).
`timescale 1ns/1ps
module pool_store_writer #(
    parameter integer IMG_W = 800,
    parameter integer IMG_H = 800,
    parameter QROM_HEX = "qrom.hex",
    parameter integer AW = 17
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          pixel_in_valid,
    input  wire [7:0]    pixel_in,
    output reg           ps_we,
    output reg  [AW-1:0] ps_addr,
    output reg  [7:0]    ps_wdata,
    output reg           frame_in_done
);
    localparam integer HALF_W = IMG_W / 2;

    // quantiser ROM (1-cycle registered read)
    reg [7:0] qrom [0:255];
    initial $readmemh(QROM_HEX, qrom);
    reg [7:0] q;
    reg       v1;
    always @(posedge clk) begin
        q  <= qrom[pixel_in];
        v1 <= pixel_in_valid & rstn;
    end

    reg [15:0] cx, cy;               // coordinates of the pixel currently in stage 1
    reg [7:0]  q_prev;
    reg [15:0] rbw [0:HALF_W-1];     // row buffer: pairs of even-row q8 values
    reg [15:0] pair_q;
    reg [AW-1:0] wr_ptr;
    wire odd_x = cx[0];
    wire odd_y = cy[0];
    wire [9:0] s_sum = pair_q[15:8] + pair_q[7:0] + q_prev + q;   // <= 4*255
    wire [7:0] pooled = (s_sum + 10'd2) >> 2;

    always @(posedge clk) begin
        if (!rstn) begin
            cx <= 16'd0; cy <= 16'd0; ps_we <= 1'b0; ps_addr <= {AW{1'b0}}; ps_wdata <= 8'd0;
            frame_in_done <= 1'b0; q_prev <= 8'd0; wr_ptr <= {AW{1'b0}};
        end else begin
            ps_we <= 1'b0;
            frame_in_done <= 1'b0;
            if (v1) begin
                q_prev <= q;
                if (!odd_y) begin
                    if (odd_x) rbw[cx >> 1] <= {q_prev, q};
                end else begin
                    pair_q <= rbw[cx >> 1];       // synchronous read; the even-column read is consumed at the odd column
                    if (odd_x) begin
                        ps_we    <= 1'b1;
                        ps_addr  <= wr_ptr;
                        ps_wdata <= pooled;
                        wr_ptr   <= wr_ptr + 1'b1;
                    end
                end
                if (cx == IMG_W - 1) begin
                    cx <= 16'd0;
                    if (cy == IMG_H - 1) begin
                        cy <= 16'd0;
                        frame_in_done <= 1'b1;
                        wr_ptr <= {AW{1'b0}};
                    end else cy <= cy + 1'b1;
                end else cx <= cx + 1'b1;
            end
        end
    end
endmodule
