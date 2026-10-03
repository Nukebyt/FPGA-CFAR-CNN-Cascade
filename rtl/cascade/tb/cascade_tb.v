// cascade_tb.v -- system test for cascade_top. Streams one image, then dumps (for the Python reference check
// in check_cascade.py): the Weibull detect/x/c1 stream, the trigger events, the CNN results and the pooled store.
`timescale 1ns/1ps
`ifndef Q4V
 `define Q4V 0
`endif
`ifndef IMGW
 `define IMGW 128
`endif
`ifndef IMGH
 `define IMGH 128
`endif
module cascade_tb;
    localparam integer IMG_W = `IMGW, IMG_H = `IMGH;
    localparam integer N_PIX = IMG_W * IMG_H;
    reg clk = 0, rstn = 0;
    always #10 clk = ~clk;                       // 50 MHz

    reg         pixel_in_valid = 0;
    reg  [7:0]  pixel_in = 0;
    reg  [1:0]  pfa_sel = 2'd0;                  // 0 -> Pfa = 1e-3
    reg  signed [18:0] tau_q14 = 19'sd`TAU;
    reg  signed [31:0] theta = -32'sd`THETA;
    wire        ready_for_frame, res_valid, res_accept, all_done, fifo_overflow;
    wire [9:0]  res_j, res_i;
    wire signed [31:0] res_logit;
    wire [15:0] n_events, n_accepted;

    cascade_top #(.IMG_W(IMG_W), .IMG_H(IMG_H), .LUT_ROOT(`LUTROOT), .QROM_HEX(`QROMHEX),
                  .CNN_W_HEX(`WHEX), .CNN_PQ_HEX(`PQHEX), .CNN_Q4(`Q4V)) dut (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in), .pfa_sel(pfa_sel),
        .tau_q14(tau_q14), .theta(theta), .ready_for_frame(ready_for_frame),
        .res_valid(res_valid), .res_j(res_j), .res_i(res_i), .res_logit(res_logit), .res_accept(res_accept),
        .all_done(all_done), .n_events(n_events), .n_accepted(n_accepted), .fifo_overflow(fifo_overflow));

    reg [7:0] pix [0:N_PIX-1];
    integer k, fdet, fev, fres, cyc_start, cyc_end, n_det;
    reg [8*160-1:0] fname;

    always @(posedge clk) begin
        if (dut.det_valid) begin $fdisplay(fdet, "%0d %0d %0d", dut.det, dut.x_o, dut.c1_o); n_det = n_det + 1; end
        if (dut.ev_valid)  $fdisplay(fev, "%0d %0d", dut.ev_j, dut.ev_i);
        if (res_valid)     $fdisplay(fres, "%0d %0d %0d %0d", res_j, res_i, res_logit, res_accept);
    end

    initial begin
        n_det = 0;
        $readmemh(`IMGHEX, pix);
        $sformat(fname, "%0s/rtl_det.txt", `OUTDIR);     fdet = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_events.txt", `OUTDIR);  fev  = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_results.txt", `OUTDIR); fres = $fopen(fname, "w");
        repeat (5) @(posedge clk);
        rstn = 1;
        repeat (3) @(posedge clk);
        while (!ready_for_frame) @(posedge clk);
        cyc_start = $time / 20;
        for (k = 0; k < N_PIX; k = k + 1) begin
            pixel_in_valid <= 1'b1; pixel_in <= pix[k];
            @(posedge clk);
        end
        pixel_in_valid <= 1'b0;
        while (!all_done) @(posedge clk);
        cyc_end = $time / 20;
        repeat (4) @(posedge clk);
        $sformat(fname, "%0s/rtl_store.hex", `OUTDIR);
        $writememh(fname, dut.store.mem);
        $fclose(fdet); $fclose(fev); $fclose(fres);
        $display("cascade_tb: %0d det pulses, %0d events, %0d accepted, overflow=%0b, %0d cycles total",
                 n_det, n_events, n_accepted, fifo_overflow, cyc_end - cyc_start);
        $finish;
    end

    initial begin
        #4000000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
