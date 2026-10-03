// isqrt_pipe.v -- integer square root floor(sqrt(x)), x < 2^16, result < 2^8, fully pipelined (one result per clock, latency 8).
// Digit-by-digit: from bit 7 down to bit 0 the candidate root|bit is kept if its square does not exceed x.
`timescale 1ns/1ps
module isqrt_pipe (
    input  wire        clk,
    input  wire        in_valid,
    input  wire [15:0] x,
    output wire        out_valid,
    output wire [7:0]  root
);
    wire [15:0] xw [0:8];
    wire [7:0]  rw [0:8];
    wire        vw [0:8];
    assign xw[0] = x; assign rw[0] = 8'd0; assign vw[0] = in_valid;
    genvar s;
    generate
        for (s = 0; s < 8; s = s + 1) begin : st
            wire [7:0]  cand = rw[s] | (8'd1 << (7 - s));
            wire [15:0] sq = cand * cand;
            reg [15:0] xr; reg [7:0] rr; reg vr;
            always @(posedge clk) begin
                xr <= xw[s]; vr <= vw[s];
                rr <= (sq <= xw[s]) ? cand : rw[s];
            end
            assign xw[s+1] = xr; assign rw[s+1] = rr; assign vw[s+1] = vr;
        end
    endgenerate
    assign out_valid = vw[8];
    assign root = rw[8];
endmodule
