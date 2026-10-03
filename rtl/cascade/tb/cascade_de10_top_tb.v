// cascade_de10_top_tb.v -- board wrapper regression: one full frame through the DE10 wrapper (shortened display
// time), logging the same files cascade_tb.v does so tb/check_cascade.py can verify the pooled store, the
// trigger+gate events and every CNN logit independently.  Also prints what the LEDs / 7-segment displays show.
`timescale 1ns/1ps
module cascade_de10_top_tb;
    reg clk = 0, key0 = 0;
    always #10 clk = ~clk;
    reg [9:0] sw = 10'b00_01_10_00;           // SW[5:4]=1 (85% point)  SW[3:2]=2 (tau 0.75)  SW[1:0]=0 (Pfa 1e-3)
    wire [9:0] ledr;
    wire [6:0] hex0, hex1, hex2, hex3, hex4, hex5;

    cascade_de10_top #(.DISPLAY_CYCLES(2000), .IMG_HEX(`IMGHEX), .CNN_W_HEX(`WHEX), .CNN_PQ_HEX(`PQHEX), .THETA1(`THETA1), .CNN_Q4(`Q4V)) dut (
        .CLOCK_50(clk), .KEY({1'b1, key0}), .SW(sw), .LEDR(ledr),
        .HEX0(hex0), .HEX1(hex1), .HEX2(hex2), .HEX3(hex3), .HEX4(hex4), .HEX5(hex5));

    integer fdet, fev, fres, n_det, nframes;
    reg [8*160-1:0] fname;
    always @(posedge clk) begin
        if (dut.core.det_valid) begin $fdisplay(fdet, "%0d %0d %0d", dut.core.det, dut.core.x_o, dut.core.c1_o); n_det = n_det + 1; end
        if (dut.core.ev_valid)  $fdisplay(fev, "%0d %0d", dut.core.ev_j, dut.core.ev_i);
    end
    always @(posedge dut.clk_cnn)
        if (dut.core.res_valid) $fdisplay(fres, "%0d %0d %0d %0d", dut.core.res_j, dut.core.res_i, dut.core.res_logit, dut.core.res_accept);
    initial begin
        n_det = 0; nframes = 0;
        $sformat(fname, "%0s/rtl_det.txt", `OUTDIR);     fdet = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_events.txt", `OUTDIR);  fev  = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_results.txt", `OUTDIR); fres = $fopen(fname, "w");
        repeat (5) @(posedge clk);
        key0 = 1;
        // wait for the first frame to finish (frame pulse on LEDR[1])
        while (!ledr[1]) @(posedge clk);
        repeat (4) @(posedge clk);
        $sformat(fname, "%0s/rtl_store.hex", `OUTDIR);
        $writememh(fname, dut.core.store.mem);
        $fclose(fdet); $fclose(fev); $fclose(fres);
        $display("wrapper: %0d det pulses; candidate events (HEX5..3) = %h%h%h, accepted (HEX2..0) = %h%h%h; LEDR=%b",
                 n_det, dut.ev_shown[11:8], dut.ev_shown[7:4], dut.ev_shown[3:0],
                 dut.acc_shown[11:8], dut.acc_shown[7:4], dut.acc_shown[3:0], ledr);
        $finish;
    end
    integer cyc = 0;
    always @(posedge clk) begin
        cyc = cyc + 1;
        if (cyc % 100000 == 0)
            $display("t=%0d cyc: wrapper st=%0d  src sst=%0d  cnn cst=%0d  n_ev=%0d  pf_busy=%b  n_acc_c=%0d",
                     cyc, dut.st, dut.core.sst, dut.core.cst, dut.core.n_ev_cnt, dut.core.pf_busy, dut.core.n_acc_c); $fflush;
    end
    initial begin #6000000000; $display("TIMEOUT"); $finish; end
endmodule
