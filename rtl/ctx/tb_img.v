`timescale 1ns/1ps
module tb_img;
    reg clk = 0; always #5 clk = ~clk;
    reg rstn = 0, start = 0; reg [27:0] sq; reg [35:0] sq2; reg [19:0] cb, cd;
    wire [7:0] f4, f5, f6, f7; wire done;
    img_codes dut (.clk(clk), .rstn(rstn), .start(start), .sumq(sq), .sumq2(sq2), .cnt_b(cb), .cnt_d(cd), .f4(f4), .f5(f5), .f6(f6), .f7(f7), .done(done));
    integer fi, fo, rc; reg [63:0] a, b, c, d;
    initial begin
        fi = $fopen("img_in.txt", "r"); fo = $fopen("img_out.txt", "w");
        repeat (3) @(posedge clk); rstn = 1;
        while (!$feof(fi)) begin
            rc = $fscanf(fi, "%d %d %d %d\n", a, b, c, d);
            if (rc == 4) begin
                @(negedge clk); sq = a; sq2 = b; cb = c; cd = d; start = 1; @(negedge clk); start = 0;
                while (!done) @(posedge clk);
                $fdisplay(fo, "%0d %0d %0d %0d", f4, f5, f6, f7);
            end
        end
        $fclose(fo); $finish;
    end
endmodule
