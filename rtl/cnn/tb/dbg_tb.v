`timescale 1ns/1ps
module dbg_tb;
    localparam NPIX=1024;
    reg clk=0, rstn=0; always #10 clk=~clk;
    reg in_valid=0; reg [7:0] in_data=0; wire ready, done, detect; wire signed [31:0] logit; reg signed [31:0] theta=0;
    cnn_core #(.W_HEX(`WHEX), .PQ_HEX(`PQHEX)) dut(.clk(clk),.rstn(rstn),.in_valid(in_valid),.in_data(in_data),.ready(ready),.done(done),.logit(logit),.theta(theta),.detect(detect));
    reg [8*NPIX-1:0] gin [0:0];
    integer i, k, sum;
    reg [3:0] last_lyr;
    initial begin
        $readmemh(`GIN, gin);
        repeat(5) @(posedge clk); rstn=1; repeat(3) @(posedge clk);
        for (i=0;i<NPIX;i=i+1) begin in_valid<=1; in_data<=gin[0][8*(NPIX-1-i) +: 8]; @(posedge clk); end
        in_valid<=0;
        last_lyr = 0;
        while (!done) begin
            @(posedge clk);
            if (dut.st == 4'd2 /*S_LCFG*/ && dut.lyr != last_lyr+0 || (dut.st==4'd2 && dut.lyr!=0 && last_lyr==0)) begin
                // layer (lyr-1) just finished: dump its output buffer
                sum = 0;
                $write("after layer %0d (buf %s): first16:", dut.lyr-1, ((dut.lyr-1)%2==0)?"B":"A");
                for (k=0;k<16;k=k+1) $write(" %0d", ((dut.lyr-1)%2==0)? dut.memB[k] : dut.memA[k]);
                $write("\n");
                last_lyr = dut.lyr;
            end
        end
        $display("logit=%0d", logit);
        $finish;
    end
endmodule
