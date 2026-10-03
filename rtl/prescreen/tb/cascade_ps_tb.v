// cascade_ps_tb.v -- system test of cascade_ps_2clk (pooled prescreen + CNN): stream domain 50 MHz, CNN domain 100 MHz (behavioural PLL).
// Logs events (clk_cnn), results and the pooled store; tb/check_cascade_ps.py verifies all three against the Python models.
`timescale 1ns/1ps
module cascade_ps_tb;
    localparam integer IMG_W = `IMGW, IMG_H = `IMGH;
    localparam integer N_PIX = IMG_W * IMG_H;
    reg clk = 0, rstn = 0;
    always #10 clk = ~clk;
    wire clk_cnn, locked;
    cnn_pll pll (.refclk(clk), .rst(1'b0), .outclk(clk_cnn), .locked(locked));
    reg         pixel_in_valid = 0;
    reg  [7:0]  pixel_in = 0;
    reg  [1:0]  pfa_sel = 2'd`PFA;
    reg  [16:0] g_th = 17'd`GTH;
    reg  signed [31:0] theta = -32'sd`THETA;
    wire        ready_for_frame, frame_done, res_valid, res_accept, ev_overflow;
    wire [9:0]  res_j, res_i;
    wire signed [31:0] res_logit;
    wire [15:0] n_events, n_accepted; wire [31:0] ps_cycles;
    cascade_ps_2clk #(.IMG_W(IMG_W), .IMG_H(IMG_H), .SLI(`SLI), .GUARD(`GUARD), .QROM_HEX(`QROMHEX), .CNN_W_HEX(`WHEX), .CNN_PQ_HEX(`PQHEX), .CNN_Q4(`Q4V),
                      .KC0(`KC0), .KC1(`KC1), .KC2(`KC2), .KC3(`KC3), .NUM_LO(`NUM_LO), .NUM_HI(`NUM_HI), .EV_AW(`EVAW)) dut (
        .clk(clk), .clk_cnn(clk_cnn), .rstn(rstn & locked), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in), .pfa_sel(pfa_sel),
        .g_th(g_th), .theta(theta), .ready_for_frame(ready_for_frame), .frame_done(frame_done),
        .n_events(n_events), .n_accepted(n_accepted), .ev_overflow(ev_overflow), .ps_cycles(ps_cycles),
        .res_valid(res_valid), .res_j(res_j), .res_i(res_i), .res_logit(res_logit), .res_accept(res_accept));

    reg [7:0] pix [0:N_PIX-1];
    integer k, fev, fres, t_start, t_end, nres, sumlog, nres1, sumlog1, ev1, acc1;
    reg [8*160-1:0] fname;
    always @(posedge clk_cnn) begin
        if (dut.pe_valid) $fdisplay(fev, "%0d %0d %0d %0d %0d %0d", dut.pe_j, dut.pe_i, dut.pe_cd, dut.pe_pl[43:27], dut.pe_pl[26:8], dut.pe_pl[7:0]);
        if (res_valid) begin $fdisplay(fres, "%0d %0d %0d %0d", res_j, res_i, res_logit, res_accept); sumlog = sumlog + res_logit; nres = nres + 1; end
    end
    initial begin
        sumlog = 0; nres = 0;
        $readmemh(`IMGHEX, pix);
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
            if (`GAPS && (k % 7 == 3)) begin pixel_in_valid <= 1'b0; repeat (3) @(posedge clk); end     // irregular host: gaps in the stream
        end
        pixel_in_valid <= 1'b0;
        while (!frame_done) @(posedge clk);
        t_end = $time;
        repeat (4) @(posedge clk);
        $sformat(fname, "%0s/rtl_store.hex", `OUTDIR);
        $writememh(fname, dut.store.mem);
        $fclose(fev); $fclose(fres);
        $display("cascade_ps_tb: %0d events, %0d accepted, overflow=%0b, prescreen %0d clk_cnn cycles, frame time %0d ns (%0d cycles @50MHz)",
                 n_events, n_accepted, ev_overflow, ps_cycles, t_end - t_start, (t_end - t_start) / 20);
`ifdef ONE
        $finish;
`endif
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
