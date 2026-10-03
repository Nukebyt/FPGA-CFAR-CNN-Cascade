// cascade_tb2.v -- two-clock system test: stream domain 50 MHz, CNN domain 100 MHz (behavioural PLL model, async phase).
// Logs the same files as cascade_tb.v so tb/check_cascade.py can verify store / events / logits independently.
`timescale 1ns/1ps
`ifndef IMGW
 `define IMGW 128
`endif
`ifndef IMGH
 `define IMGH 128
`endif
module cascade_tb2;
    localparam integer IMG_W = `IMGW, IMG_H = `IMGH;
    localparam integer N_PIX = IMG_W * IMG_H;
    reg clk = 0, rstn = 0;
    always #10 clk = ~clk;                        // 50 MHz
    wire clk_cnn, locked;
    cnn_pll pll (.refclk(clk), .rst(1'b0), .outclk(clk_cnn), .locked(locked));

    reg         pixel_in_valid = 0;
    reg  [7:0]  pixel_in = 0;
    reg  [1:0]  pfa_sel = 2'd0;
    reg  signed [18:0] tau_q14 = 19'sd`TAU;
    reg  signed [31:0] theta = -32'sd`THETA;
    wire        ready_for_frame, frame_done, res_valid, res_accept, ev_overflow;
    wire [9:0]  res_j, res_i;
    wire signed [31:0] res_logit;
    wire [15:0] n_events, n_accepted;

    cascade_top_2clk #(.IMG_W(IMG_W), .IMG_H(IMG_H), .LUT_ROOT(`LUTROOT), .QROM_HEX(`QROMHEX),
                       .CNN_W_HEX(`WHEX), .CNN_PQ_HEX(`PQHEX), .CNN_Q4(`Q4V)) dut (
        .clk(clk), .clk_cnn(clk_cnn), .rstn(rstn & locked), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in), .pfa_sel(pfa_sel),
        .tau_q14(tau_q14), .theta(theta), .ready_for_frame(ready_for_frame), .frame_done(frame_done),
        .n_events(n_events), .n_accepted(n_accepted), .ev_overflow(ev_overflow),
        .res_valid(res_valid), .res_j(res_j), .res_i(res_i), .res_logit(res_logit), .res_accept(res_accept));

    reg [7:0] pix [0:N_PIX-1];
    integer k, fdet, fev, fres, n_det, t_start, t_end;
    integer sumlog, nres, sumlog1, nres1, ev1, acc1;
    reg [8*160-1:0] fname;
    always @(posedge clk) begin
        if (dut.det_valid) begin $fdisplay(fdet, "%0d %0d %0d", dut.det, dut.x_o, dut.c1_o); n_det = n_det + 1; end
        if (dut.ev_valid)  $fdisplay(fev, "%0d %0d", dut.ev_j, dut.ev_i);
    end
    always @(posedge clk_cnn) if (res_valid) begin
        $fdisplay(fres, "%0d %0d %0d %0d", res_j, res_i, res_logit, res_accept);
        sumlog = sumlog + res_logit; nres = nres + 1;
    end

    initial begin
        n_det = 0; sumlog = 0; nres = 0;
        $readmemh(`IMGHEX, pix);
        $sformat(fname, "%0s/rtl_det.txt", `OUTDIR);     fdet = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_events.txt", `OUTDIR);  fev  = $fopen(fname, "w");
        $sformat(fname, "%0s/rtl_results.txt", `OUTDIR); fres = $fopen(fname, "w");
        repeat (5) @(posedge clk);
        rstn = 1;
        repeat (6) @(posedge clk);
        while (!ready_for_frame) @(posedge clk);
        t_start = $time;
        for (k = 0; k < N_PIX; k = k + 1) begin
            pixel_in_valid <= 1'b1; pixel_in <= pix[k];
            @(posedge clk);
        end
        pixel_in_valid <= 1'b0;
        while (!frame_done) @(posedge clk);
        t_end = $time;
        repeat (4) @(posedge clk);
        $sformat(fname, "%0s/rtl_store.hex", `OUTDIR);
        $writememh(fname, dut.store.mem);
        $fclose(fdet); $fclose(fev); $fclose(fres);
        $display("cascade_tb2: %0d det pulses, %0d events, %0d accepted, overflow=%0b, frame time %0d ns (%0d cycles @50MHz)",
                 n_det, n_events, n_accepted, ev_overflow, t_end - t_start, (t_end - t_start) / 20);
        sumlog1 = sumlog; nres1 = nres; ev1 = n_events; acc1 = n_accepted;
        // second frame through the same hardware: the handshake/reset path must leave it reusable
        while (!ready_for_frame) @(posedge clk);
        sumlog = 0; nres = 0;
        for (k = 0; k < N_PIX; k = k + 1) begin
            pixel_in_valid <= 1'b1; pixel_in <= pix[k];
            @(posedge clk);
        end
        pixel_in_valid <= 1'b0;
        while (!frame_done) @(posedge clk);
        repeat (4) @(posedge clk);
        $display("frame 2: %0d events, %0d accepted, %0d results, sum(logit)=%0d   | frame 1: %0d events, %0d accepted, %0d results, sum(logit)=%0d",
                 n_events, n_accepted, nres, sumlog, ev1, acc1, nres1, sumlog1);
        if (n_events == ev1 && n_accepted == acc1 && nres == nres1 && sumlog == sumlog1) $display("SECOND FRAME IDENTICAL: PASS");
        else $display("SECOND FRAME MISMATCH: FAIL");
        $finish;
    end
    initial begin #4000000000; $display("TIMEOUT"); $finish; end
endmodule
