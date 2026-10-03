// weibull_delta_rom.v -- 16384-entry ROM, address = {pfa_sel, c2_addr}
module weibull_delta_rom #(
    parameter HEX_FILE = "lut/weibull/weibull_delta_lut.hex",
    parameter integer DW = 16
) (
    input  wire        clk,
    input  wire [1:0]  pfa_sel,
    input  wire [11:0] c2_addr,
    output reg  signed [DW-1:0] dout
);
    reg signed [DW-1:0] mem [0:16383];
    initial $readmemh(HEX_FILE, mem);
    wire [13:0] flat = {pfa_sel, c2_addr};
    always @(posedge clk) dout <= mem[flat];
endmodule
