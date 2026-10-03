// cnn_core_tb.v -- runs the exported golden vectors (cnn/quant_hw.py --export) through cnn_core and
// compares every integer logit with the Python integer-exact reference.
//
//   iverilog -g2005 -I<gen_dir> -DWHEX=\"<gen_dir>/cnn_w.hex\" -DPQHEX=\"<gen_dir>/cnn_pq.hex\" \
//            -DGIN=\"<export>/golden_in.hex\" -DGLG=\"<export>/golden_logit_int.txt\" -DNV=16 \
//            -o sim.vvp cnn_core.v tb/cnn_core_tb.v && vvp sim.vvp
`timescale 1ns/1ps
`ifndef NV
 `define NV 16
`endif
module cnn_core_tb;
    localparam integer NV = `NV;
    localparam integer NPIX = 1024;
    reg clk = 0, rstn = 0;
    always #10 clk = ~clk;          // 50 MHz

    reg         in_valid = 0;
    reg  [7:0]  in_data = 0;
    wire        ready, done, detect;
    wire signed [31:0] logit;
    reg  signed [31:0] theta = 0;

`ifndef CORE
 `define CORE cnn_core
`endif
    `CORE #(.W_HEX(`WHEX), .PQ_HEX(`PQHEX)) dut (
        .clk(clk), .rstn(rstn), .in_valid(in_valid), .in_data(in_data), .ready(ready),
        .done(done), .logit(logit), .theta(theta), .detect(detect));

    reg [8*NPIX-1:0] gin [0:NV-1];
    integer exp_logit [0:NV-1];
    integer fh, i, v, errs, rc, cyc0, cyc1;
    integer total_cyc;

    initial begin
        $readmemh(`GIN, gin);
        fh = $fopen(`GLG, "r");
        for (i = 0; i < NV; i = i + 1) rc = $fscanf(fh, "%d\n", exp_logit[i]);
        $fclose(fh);
        errs = 0; total_cyc = 0;
        repeat (5) @(posedge clk);
        rstn = 1;
        repeat (3) @(posedge clk);
        for (v = 0; v < NV; v = v + 1) begin
            while (!ready) @(posedge clk);
            // stream the patch, one byte per clock (first byte = leftmost hex digits)
            for (i = 0; i < NPIX; i = i + 1) begin
                in_valid <= 1; in_data <= gin[v][8*(NPIX-1-i) +: 8];
                @(posedge clk);
            end
            in_valid <= 0;
            cyc0 = $time / 20;
            while (!done) @(posedge clk);
            cyc1 = $time / 20;
            total_cyc = total_cyc + (cyc1 - cyc0);
            if (logit !== exp_logit[v]) begin
                errs = errs + 1;
                if (errs <= 10) $display("MISMATCH vec %0d: rtl=%0d expected=%0d", v, logit, exp_logit[v]);
            end
            @(posedge clk);
        end
        $display("cnn_core_tb: %0d vectors, %0d mismatches, %0d compute cycles/patch (avg)",
                 NV, errs, total_cyc / NV);
        if (errs == 0) $display("PASS");
        else $display("FAIL");
        $finish;
    end

    initial begin
        #2000000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
