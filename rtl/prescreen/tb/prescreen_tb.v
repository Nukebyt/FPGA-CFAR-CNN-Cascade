// prescreen_tb.v -- prescreen_top against the Python golden model (tb/check_prescreen.py).
// Runs the frame twice back to back (second pass checks that every state is re-initialised by `start`); events of the
// two passes go to OUTFILE and OUTFILE2.   Macros: WP HP SLI GUARD KC0..KC3 NUM_LO NUM_HI PFA GTH PHEX OUTFILE OUTFILE2 MAXCYC
`timescale 1ns/1ps
module prescreen_tb;
    reg clk = 0, rstn = 0, start = 0;
    always #5 clk = ~clk;
    localparam integer WPv = `WP, HPv = `HP;
    reg [7:0] mem [0:WPv*HPv-1];
    initial $readmemh(`PHEX, mem);
    wire [17:0] raddr; reg [7:0] rdata;
    always @(posedge clk) rdata <= mem[raddr];
    wire busy, ev_valid, done; wire [9:0] ev_j, ev_i; wire [16:0] ev_cd; wire [43:0] ev_pl;
    prescreen_top #(.WP(WPv), .HP(HPv), .SLI(`SLI), .GUARD(`GUARD), .KC0(`KC0), .KC1(`KC1), .KC2(`KC2), .KC3(`KC3),
                    .NUM_LO(`NUM_LO), .NUM_HI(`NUM_HI)) dut (
        .clk(clk), .rstn(rstn), .start(start), .pfa_sel(2'd`PFA), .g_th(17'd`GTH), .ps_raddr(raddr), .ps_rdata(rdata),
        .busy(busy), .ev_valid(ev_valid), .ev_j(ev_j), .ev_i(ev_i), .ev_cd(ev_cd), .ev_pl(ev_pl), .done(done));
    integer f1, f2, fcur, n, cyc, pass;
    task run_frame;
        begin
            n = 0; cyc = 0;
            @(negedge clk); start = 1; @(negedge clk); start = 0;
            while (!done) begin
                @(posedge clk); cyc = cyc + 1;
                if (cyc > `MAXCYC) begin $display("TIMEOUT"); $finish; end
            end
            repeat (4) @(posedge clk);
            $display("prescreen_tb pass %0d: %0d events, %0d clocks until done", pass, n, cyc);
        end
    endtask
    initial begin
        f1 = $fopen(`OUTFILE, "w"); f2 = $fopen(`OUTFILE2, "w"); fcur = f1; pass = 1;
        repeat (4) @(posedge clk); rstn = 1; repeat (4) @(posedge clk);
        run_frame;
        fcur = f2; pass = 2; repeat (20) @(posedge clk);
        run_frame;
        $fclose(f1); $fclose(f2);
        $finish;
    end
    always @(posedge clk) if (ev_valid) begin
        n = n + 1;
        // pl = {S1[16:0], numsh[18:0], P[7:0]}
        $fdisplay(fcur, "%0d %0d %0d %0d %0d %0d", ev_j, ev_i, ev_cd, ev_pl[43:27], ev_pl[26:8], ev_pl[7:0]);
    end
endmodule
