`timescale 1ns/1ps
module tb_side;
    reg clk = 0; always #5 clk = ~clk;
    reg v = 0; reg [16:0] a; reg [16:0] s1; reg [18:0] n; reg [7:0] p;
    wire ov; wire [31:0] codes;
    side_unit dut (.clk(clk), .in_valid(v), .a(a), .s1(s1), .numsh(n), .p(p), .out_valid(ov), .codes(codes));
    integer fi, fo, rc, ai, si, ni, pi;
    initial begin
        fi = $fopen("side_in.txt", "r"); fo = $fopen("side_out.txt", "w");
        repeat (3) @(posedge clk);
        while (!$feof(fi)) begin
            rc = $fscanf(fi, "%d %d %d %d\n", ai, si, ni, pi);
            if (rc == 4) begin @(negedge clk); v = 1; a = ai; s1 = si; n = ni; p = pi; end
        end
        @(negedge clk); v = 0;
        repeat (20) @(posedge clk);
        $fclose(fo); $finish;
    end
    always @(posedge clk) if (ov) $fdisplay(fo, "%0d %0d %0d %0d", codes[7:0], codes[15:8], codes[23:16], codes[31:24]);
endmodule
