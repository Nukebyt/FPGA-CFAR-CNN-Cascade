// log_amp_rom.v
// 256-entry ROM: 8-bit pixel value -> centred log-amplitude xc (Q1.14 signed).
// Contents come verbatim from lut/shared/log_amp_lut.hex (generate_shared_luts.m):
// xc = log(sqrt(max(I,IFloor)+0.5)) - X0, X0=1.2125 baked in at generation time.
module log_amp_rom #(
    parameter HEX_FILE = "lut/shared/log_amp_lut.hex"
) (
    input  wire        clk,
    input  wire [7:0]  addr,
    output reg  signed [15:0] dout
);
    reg signed [15:0] mem [0:255];
    initial $readmemh(HEX_FILE, mem);
    always @(posedge clk) dout <= mem[addr];
endmodule
