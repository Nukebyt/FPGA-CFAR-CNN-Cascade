// event_ram_dc.v -- candidate-event store: written (in order) in the stream domain during pass 1, read in the CNN
// domain during pass 2. Replaces the single-clock FIFO: the read side only starts after the go handshake, when the
// write pointer is quiescent, so a plain dual-clock RAM + a quasi-static count is clock-domain-crossing safe.
`timescale 1ns/1ps
module event_ram_dc #(
    parameter integer WIDTH = 20,
    parameter integer AW    = 10
) (
    input  wire             wclk,
    input  wire             we,
    input  wire [AW-1:0]    waddr,
    input  wire [WIDTH-1:0] wdata,
    input  wire             rclk,
    input  wire [AW-1:0]    raddr,
    output reg  [WIDTH-1:0] rdata
);
    reg [WIDTH-1:0] mem [0:(1<<AW)-1];
    always @(posedge wclk) if (we) mem[waddr] <= wdata;
    always @(posedge rclk) rdata <= mem[raddr];
endmodule
