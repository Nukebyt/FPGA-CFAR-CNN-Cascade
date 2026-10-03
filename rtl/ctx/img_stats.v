// img_stats.v -- image-level statistics accumulated while the frame streams in (clk domain, gap tolerant):
//   sumQ = sum of QROM(pixel), sumQ2 = sum of QROM(pixel)^2, cntB = #(pixel > 200), cntD = #(pixel < 5)
// Reset with the frame (core_rstn).  Values are quasi-static after the frame's last pixel (read by img_codes in the CNN clock domain).
`timescale 1ns/1ps
module img_stats #(parameter QROM_HEX = "qrom.hex") (
    input  wire        clk,
    input  wire        rstn,
    input  wire        pixel_in_valid,
    input  wire [7:0]  pixel_in,
    output reg  [27:0] sumq,
    output reg  [35:0] sumq2,
    output reg  [19:0] cnt_b,
    output reg  [19:0] cnt_d
);
    reg [7:0] qrom [0:255];
    initial $readmemh(QROM_HEX, qrom);
    reg [7:0] q, pix1; reg v1;
    always @(posedge clk) begin
        q <= qrom[pixel_in]; pix1 <= pixel_in; v1 <= pixel_in_valid & rstn;
        if (!rstn) begin sumq <= 28'd0; sumq2 <= 36'd0; cnt_b <= 20'd0; cnt_d <= 20'd0; end
        else if (v1) begin
            sumq  <= sumq + q;
            sumq2 <= sumq2 + q * q;
            if (pix1 > 8'd200) cnt_b <= cnt_b + 1'b1;
            if (pix1 < 8'd5)   cnt_d <= cnt_d + 1'b1;
        end
    end
endmodule
