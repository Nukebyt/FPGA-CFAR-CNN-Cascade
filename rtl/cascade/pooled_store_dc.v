// pooled_store_dc.v -- dual-clock simple dual-port byte RAM: written in the stream clock domain, read in the CNN domain.
// The two sides never overlap in time (the CNN pass starts only after the frame has been written and the
// go/fin handshake has crossed), so no data-level synchronisation is needed.
`timescale 1ns/1ps
module pooled_store_dc #(
    parameter integer DEPTH = 160000,
    parameter integer AW    = 18
) (
    input  wire          wclk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [7:0]    wdata,
    input  wire          rclk,
    input  wire [AW-1:0] raddr,
    output reg  [7:0]    rdata
);
    reg [7:0] mem [0:DEPTH-1];
    always @(posedge wclk) if (we) mem[waddr] <= wdata;
    always @(posedge rclk) rdata <= mem[raddr];
endmodule
