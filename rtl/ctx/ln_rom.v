// ln_rom.v -- f8 = min(255, round(25*ln(1+nE))) for nE = 0..8191 (registered read, 1 clock). Contents: side_fx.LN (ln_rom.hex)
`timescale 1ns/1ps
module ln_rom #(parameter HEX = "ln_rom.hex") (
    input  wire        clk,
    input  wire [12:0] n,
    output reg  [7:0]  code
);
    reg [7:0] mem [0:8191];
    initial $readmemh(HEX, mem);
    always @(posedge clk) code <= mem[n];
endmodule
