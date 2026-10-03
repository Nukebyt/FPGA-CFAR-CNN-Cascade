// cascade_ps_jtag_tb.v -- drives cascade_ps_jtag_core exactly like the JTAG master will: configure, START, push 32-bit pixel
// words with random idle gaps (the Weibull core must still see a gap-free stream), poll status, read results back.
// Also dumps the same internal logs as the other cascade TBs so tb/check_cascade.py can verify the run independently.
`timescale 1ns/1ps
module cascade_ps_jtag_tb;
    localparam integer W = `IMGW, H = `IMGH;
    localparam integer NWORDS = W * H / 4;
    reg clk = 0, rstn = 0;
    always #10 clk = ~clk;
    wire clk_cnn, locked;
    cnn_pll pll (.refclk(clk), .rst(1'b0), .outclk(clk_cnn), .locked(locked));

    reg [15:0] a = 0; reg win = 0; reg wr = 0, rd = 0; reg [31:0] wd = 0;
    wire [31:0] rdata; wire rdv, wait_r;
    wire st_ready, st_loading, st_done, st_ovf; wire [15:0] st_nev, st_nacc;
    cascade_ps_jtag_core #(.IMG_W(W), .IMG_H(H), .QROM_HEX("F:/Projects/CFAR/rtl/cascade/qrom.hex"),
                        .CNN_W_HEX(`WHEX), .CNN_PQ_HEX(`PQHEX), .CNN_Q4(`Q4V)) dut (
        .clk(clk), .clk_cnn(clk_cnn), .rstn_in(rstn), .pll_locked(locked),
        .avs_address(a), .avs_win(win), .avs_write(wr), .avs_writedata(wd), .avs_read(rd), .avs_readdata(rdata),
        .avs_readdatavalid(rdv), .avs_waitrequest(wait_r),
        .st_ready(st_ready), .st_loading(st_loading), .st_done(st_done), .st_ovf(st_ovf), .st_n_events(st_nev), .st_n_accepted(st_nacc));

    task avs_wr(input [15:0] ad, input [31:0] d);
        begin
            @(posedge clk); #1 a = ad; win = 0; wd = d; wr = 1;
            @(posedge clk); while (wait_r) @(posedge clk);
            #1 wr = 0;
        end
    endtask
    reg [31:0] rv;
    task avs_rd(input [15:0] ad);
        begin
            @(posedge clk); #1 a = ad; rd = 1;
            @(posedge clk); #1 rd = 0;
            while (!rdv) @(posedge clk);
            rv = rdata;
        end
    endtask

    reg [7:0] img [0:W*H-1];
    integer i, k, gap, seed, nev, nacc, nres, fres, fev;
    reg [31:0] lo, hi;
    reg [8*160-1:0] fname;
    // internal logs (clk_cnn domain: the prescreen runs there)
    always @(posedge clk_cnn)
        if (dut.core.pe_valid) $fdisplay(fev, "%0d %0d %0d %0d %0d %0d", dut.core.pe_j, dut.core.pe_i, dut.core.pe_cd, dut.core.pe_pl[43:27], dut.core.pe_pl[26:8], dut.core.pe_pl[7:0]);

    task run_frame(input integer frame_no, input integer maxgap);
        begin
            avs_rd(16'h0005);
            while (!(rv[0])) begin repeat (20) @(posedge clk); avs_rd(16'h0005); end      // ready
            avs_wr(16'h0001, 32'd1);                                                       // START_FRAME
            for (i = 0; i < NWORDS; i = i + 1) begin
                @(posedge clk); #1 a = 16'hxxxx; win = 1; wd = {img[4*i+3], img[4*i+2], img[4*i+1], img[4*i]}; wr = 1;
                @(posedge clk); while (wait_r) @(posedge clk);
                #1 wr = 0; win = 0;
                gap = (maxgap == 0) ? 0 : ($random(seed) & 32'h7fffffff) % (maxgap + 1);
                repeat (gap) @(posedge clk);
            end
            avs_rd(16'h0005);
            while (!(rv[2])) begin repeat (200) @(posedge clk); avs_rd(16'h0005); end      // frame_done
            avs_rd(16'h0006); nev = rv;
            avs_rd(16'h0007); nacc = rv;
            avs_rd(16'h0008); nres = rv;
            $display("frame %0d: n_events=%0d n_accepted=%0d n_res=%0d", frame_no, nev, nacc, nres);
            avs_rd(16'h0009); $display("   pixels issued = %0d (expect %0d)", rv, W * H);
            avs_rd(16'h000A); $display("   post-stream cycles = %0d", rv);
            avs_rd(16'h000B); $display("   prescreen clk_cnn cycles = %0d", rv);
            avs_rd(16'h0005); $display("   status = %b", rv[5:0]);
            for (k = 0; k < nres; k = k + 1) begin
                avs_rd(16'h1000 + 2 * k);     lo = rv;
                avs_rd(16'h1001 + 2 * k);     hi = rv;
                $fdisplay(fres, "%0d %0d %0d %0d", lo[19:10], lo[9:0], $signed(hi), lo[29]);
            end
        end
    endtask

    initial begin
        $readmemh(`IMGHEX, img);
        seed = 12345;
                $sformat(fname, "%0s/rtl_events.txt", `OUTDIR);  fev  = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_results.txt", `OUTDIR); fres = $fopen(fname, "w");
        repeat (5) @(posedge clk); rstn = 1; repeat (10) @(posedge clk);
        avs_rd(16'h0000); $display("ID = %h", rv);
        avs_wr(16'h0002, `PFA);                  // pfa plane
        avs_wr(16'h0003, `GTH);                  // prescreen gate
        avs_wr(16'h0004, -`THETA1);              // CNN theta
        run_frame(1, 12);
        $sformat(fname, "%0s/rtl_store.hex", `OUTDIR);
        $writememh(fname, dut.core.store.mem);
        $fclose(fev); $fclose(fres);
        if (`TWO) begin                          // reuse check: second frame, back-to-back, no gaps at all
            fres = $fopen("`OUTDIR/second_results.txt", "w");
            run_frame(2, 0);
            $fclose(fres);
        end
        $finish;
    end
    initial begin #4000000000; $display("TIMEOUT"); $finish; end
endmodule
