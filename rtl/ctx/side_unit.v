// side_unit.v -- per-event side features f0..f3 (golden model: _comparison/fixedpoint/side_fx.py event_codes), pipelined, latency 12 clocks.
//   f0 = min(255, (A*195 + 2^15) >> 16)    f1 = (S1*195 + 2^15) >> 16    f2 = isqrt((numsh*2385 + 2^15) >> 16)    f3 = P
`timescale 1ns/1ps
module side_unit (
    input  wire        clk,
    input  wire        in_valid,
    input  wire [16:0] a,
    input  wire [16:0] s1,
    input  wire [18:0] numsh,
    input  wire [7:0]  p,
    output wire        out_valid,
    output wire [31:0] codes            // {f3, f2, f1, f0}
);
    wire [33:0] prodA = a * 17'd195 + 34'd32768;
    wire [33:0] prodS = s1 * 17'd195 + 34'd32768;
    wire [31:0] prodN = numsh * 13'd2385 + 32'd32768;
    reg        v1; reg [7:0] f0_1, f1_1, p_1; reg [15:0] c2v_1;
    always @(posedge clk) begin
        v1 <= in_valid;
        f0_1 <= (prodA[33:16] > 18'd255) ? 8'd255 : prodA[23:16];
        f1_1 <= prodS[23:16];
        p_1 <= p;
        c2v_1 <= (prodN[31:16] > 16'hFFFF) ? 16'hFFFF : prodN[31:16];
    end
    wire sv; wire [7:0] sroot;
    isqrt_pipe sq (.clk(clk), .in_valid(v1), .x(c2v_1), .out_valid(sv), .root(sroot));
    // align f0, f1, p with the 8-clock root pipeline
    reg [23:0] d [0:7];
    integer i;
    always @(posedge clk) begin
        d[0] <= {p_1, f1_1, f0_1};
        for (i = 1; i < 8; i = i + 1) d[i] <= d[i-1];
    end
    assign out_valid = sv;
    assign codes = {d[7][23:16], sroot, d[7][15:8], d[7][7:0]};
endmodule
