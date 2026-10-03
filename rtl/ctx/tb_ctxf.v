`timescale 1ns/1ps
module tb_ctxf;
    reg clk = 0; always #5 clk = ~clk;
    localparam WP = 400, HP = 400;
    reg [7:0] mem [0:WP*HP-1];
    initial $readmemh("P.hex", mem);
    wire [17:0] ra; reg [7:0] rd; always @(posedge clk) rd <= mem[ra];
    reg rstn = 0, start = 0; reg [9:0] j = 0, i = 0;
    wire pv; wire [7:0] pd; wire busy;
    ctx_fetch #(.WP(WP), .HP(HP)) dut (.clk(clk), .rstn(rstn), .start(start), .j(j), .i(i), .ps_raddr(ra), .ps_rdata(rd), .px_valid(pv), .px_data(pd), .busy(busy));
    integer fo, n, t;
    reg [9:0] tj [0:5]; reg [9:0] ti [0:5];
    initial begin
        tj[0] = 0; ti[0] = 0; tj[1] = 399; ti[1] = 399; tj[2] = 200; ti[2] = 37; tj[3] = 5; ti[3] = 390; tj[4] = 330; ti[4] = 12; tj[5] = 211; ti[5] = 213;
        fo = $fopen("ctxf_out.txt", "w");
        repeat (4) @(posedge clk); rstn = 1; repeat (3) @(posedge clk);
        for (t = 0; t < 6; t = t + 1) begin
            @(negedge clk); j = tj[t]; i = ti[t]; start = 1; @(negedge clk); start = 0;
            n = 0; while (n < 1024) begin @(posedge clk); if (pv) begin $fdisplay(fo, "%0d", pd); n = n + 1; end end
            while (busy) @(posedge clk); repeat (3) @(posedge clk);
        end
        $fclose(fo); $finish;
    end
endmodule
