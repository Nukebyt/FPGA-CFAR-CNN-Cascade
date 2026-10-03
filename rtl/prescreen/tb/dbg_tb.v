`timescale 1ns/1ps
module dbg_tb;
    reg clk = 0, rstn = 0, start = 0;
    always #5 clk = ~clk;
    reg [7:0] mem [0:`WP*`HP-1];
    initial $readmemh(`PHEX, mem);
    wire [17:0] raddr; reg [7:0] rdata;
    always @(posedge clk) rdata <= mem[raddr];
    wire busy, ev_valid, done; wire [9:0] ev_j, ev_i; wire [16:0] ev_cd; wire [43:0] ev_pl;
    prescreen_top #(.WP(`WP), .HP(`HP), .SLI(`SLI), .GUARD(`GUARD), .KC0(`KC0), .KC1(`KC1), .KC2(`KC2), .KC3(`KC3), .NUM_LO(`NUM_LO), .NUM_HI(`NUM_HI)) dut (
        .clk(clk), .rstn(rstn), .start(start), .pfa_sel(2'd`PFA), .g_th(17'd`GTH), .ps_raddr(raddr), .ps_rdata(rdata),
        .busy(busy), .ev_valid(ev_valid), .ev_j(ev_j), .ev_i(ev_i), .ev_cd(ev_cd), .ev_pl(ev_pl), .done(done));
    integer nt=0, ns=0, no=0, nl=0, cyc=0;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc>=8 && cyc<30) $display("cyc %0d busy=%b s=%0d r=%0d c=%0d sv3=%b sd3=%0d raddr=%0d rdata=%0d tok_go=%b", cyc, dut.core.busy, dut.core.s, dut.core.r, dut.core.c, dut.core.sv3, dut.core.sd3, raddr, rdata, dut.core.tok_go);
        if (dut.core.tok_go) nt = nt + 1;
        if (dut.core.s_v) begin ns = ns + 1; if (ns <= 3 || ns % 500 == 0) $display("s_v #%0d cyc %0d: s1=%0d s2=%0d p=%0d", ns, cyc, dut.core.s1_o, dut.core.s2_o, dut.core.p_o); end
        if (dut.core.o_valid) begin no = no + 1; if (dut.core.o_cd != 0) $display("o_cd!=0 #%0d cd=%0d", no, dut.core.o_cd); end
        if (dut.core.o_last) $display("o_last at cyc %0d (tok %0d s_v %0d o_valid %0d)", cyc, nt, ns, no);
        if (done) $display("done cyc %0d", cyc);
    end
    initial begin
        repeat (4) @(posedge clk); rstn = 1; repeat (4) @(posedge clk);
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        repeat (`MAXCYC) @(posedge clk);
        $display("end: tokens %0d s_v %0d o_valid %0d busy=%b", nt, ns, no, busy);
        $finish;
    end
endmodule
module dbg2;
endmodule
