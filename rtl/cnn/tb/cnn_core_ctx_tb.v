// cnn_core_ctx_tb.v -- golden vectors (quant_ctx.py) through cnn_core_ctx: for every vector the three input streams (side bytes, context patch, fine patch)
// are sent in a vector-dependent ORDER (the core accepts any order) and every integer logit is compared with the numpy reference.
//   iverilog -g2012 -I<gen_dir> -DWHEX=\"..\" -DPQHEX=\"..\" -DGDIR=\"<golden dir>\" -DNV=48 -o sim.vvp cnn_core_ctx.v tb/cnn_core_ctx_tb.v
`timescale 1ns/1ps
module cnn_core_ctx_tb;
    localparam integer NV = `NV;
    localparam integer NPIX = 1024;
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;
    reg in_valid = 0; reg [7:0] in_data = 0; reg [1:0] in_kind = 0;
    wire ready, done, detect; wire signed [31:0] logit; reg signed [31:0] theta = 0;
    cnn_core_ctx #(.W_HEX(`WHEX), .PQ_HEX(`PQHEX)) dut (.clk(clk), .rstn(rstn), .in_valid(in_valid), .in_data(in_data), .in_kind(in_kind),
        .ready(ready), .done(done), .logit(logit), .theta(theta), .detect(detect));
    reg [8*NPIX-1:0] gf [0:NV-1];
    reg [8*NPIX-1:0] gc [0:NV-1];
    reg [8*9-1:0]    gs [0:NV-1];
    integer exp_logit [0:NV-1];
    integer fh, i, v, errs, rc, c0, c1, tot, order, k;
    task send_fine; begin for (i = 0; i < NPIX; i = i + 1) begin in_valid <= 1; in_kind <= 2'd0; in_data <= gf[v][8*(NPIX-1-i) +: 8]; @(posedge clk); end in_valid <= 0; end endtask
    task send_ctx;  begin for (i = 0; i < NPIX; i = i + 1) begin in_valid <= 1; in_kind <= 2'd1; in_data <= gc[v][8*(NPIX-1-i) +: 8]; @(posedge clk); end in_valid <= 0; end endtask
    task send_side; begin for (i = 0; i < 9; i = i + 1) begin in_valid <= 1; in_kind <= 2'd2; in_data <= gs[v][8*(8-i) +: 8]; @(posedge clk); end in_valid <= 0; end endtask
    initial begin
        $readmemh({`GDIR, "/gin_fine.hex"}, gf); $readmemh({`GDIR, "/gin_ctx.hex"}, gc); $readmemh({`GDIR, "/gin_side.hex"}, gs);
        fh = $fopen({`GDIR, "/golden_logit.txt"}, "r");
        for (i = 0; i < NV; i = i + 1) rc = $fscanf(fh, "%d\n", exp_logit[i]);
        $fclose(fh);
        errs = 0; tot = 0;
        repeat (5) @(posedge clk); rstn = 1; repeat (3) @(posedge clk);
        for (v = 0; v < NV; v = v + 1) begin
            while (!ready) @(posedge clk);
            order = v % 6;
            case (order)
                0: begin send_side; send_ctx; send_fine; end
                1: begin send_fine; send_ctx; send_side; end
                2: begin send_ctx; send_fine; send_side; end
                3: begin send_side; send_fine; send_ctx; end
                4: begin send_fine; send_side; send_ctx; end
                default: begin send_ctx; send_side; send_fine; end
            endcase
            c0 = $time / 10;
            while (!done) @(posedge clk);
            c1 = $time / 10; tot = tot + (c1 - c0);
            if (logit !== exp_logit[v]) begin
                errs = errs + 1;
                if (errs <= 10) $display("MISMATCH vec %0d (order %0d): rtl=%0d expected=%0d", v, order, logit, exp_logit[v]);
            end
            @(posedge clk);
        end
        $display("cnn_core_ctx_tb: %0d vectors, %0d mismatches, %0d compute cycles/candidate (avg)", NV, errs, tot / NV);
        if (errs == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
    initial begin #4000000000; $display("TIMEOUT"); $finish; end
endmodule
