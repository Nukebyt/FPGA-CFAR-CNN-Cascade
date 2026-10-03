// pooled_store.v -- simple dual-port byte RAM (registered read) holding the 2x2-pooled frame.
`timescale 1ns/1ps
module pooled_store #(
    parameter integer DEPTH = 160000,
    parameter integer AW    = 18
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [7:0]    wdata,
    input  wire [AW-1:0] raddr,
    output reg  [7:0]    rdata
);
    reg [7:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end
endmodule
