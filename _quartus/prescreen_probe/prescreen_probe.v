// resource / Fmax probe for prescreen_top at 800x800 (pooled 400x400) with a real 160 kB store behind it
module prescreen_probe (
    input  wire clk, rstn, start,
    input  wire [1:0] pfa_sel,
    input  wire [16:0] g_th,
    input  wire wr_en, input wire [17:0] wr_addr, input wire [7:0] wr_data,
    output reg  [31:0] sig, output reg busy_o, done_o, output reg [15:0] nev
);
    reg [7:0] mem [0:159999];
    reg [7:0] rdata; wire [17:0] raddr;
    always @(posedge clk) begin if (wr_en) mem[wr_addr] <= wr_data; rdata <= mem[raddr]; end
    wire busy, ev_valid, done; wire [9:0] ev_j, ev_i; wire [16:0] ev_cd; wire [43:0] ev_pl;
    prescreen_top dut (.clk(clk), .rstn(rstn), .start(start), .pfa_sel(pfa_sel), .g_th(g_th), .ps_raddr(raddr), .ps_rdata(rdata),
                       .busy(busy), .ev_valid(ev_valid), .ev_j(ev_j), .ev_i(ev_i), .ev_cd(ev_cd), .ev_pl(ev_pl), .done(done));
    always @(posedge clk) begin
        busy_o <= busy; done_o <= done;
        if (start) nev <= 16'd0; else if (ev_valid) nev <= nev + 1'b1;
        if (ev_valid) sig <= sig ^ {ev_j, ev_i, ev_cd[11:0]} ^ ev_pl[31:0] ^ {ev_pl[43:32], 20'd0};
    end
endmodule
