`timescale 1ns/1ps
module weibull_backend_tb;
    localparam integer N = 3600;
    localparam LUT_ROOT = "../../../lut";

    reg clk = 0; always #5 clk = ~clk;
    reg rstn = 0;
    reg molc_valid = 0;
    reg [25:0] c2_code = 0;
    reg [1:0] pfa_sel = 0;

    wire valid_out; wire signed [15:0] delta_out;
    wire sat_low, sat_high;

    weibull_backend_new #(.LUT_ROOT(LUT_ROOT)) dut (
        .clk(clk), .rstn(rstn), .molc_valid(molc_valid),
        .c2_code(c2_code), .pfa_sel(pfa_sel),
        .valid_out(valid_out), .delta_out(delta_out),
        .addr_sat_low(sat_low), .addr_sat_high(sat_high));

    reg [31:0] v_c2 [0:N-1];
    integer v_pfa [0:N-1];
    integer v_delta [0:N-1];
    integer i, fh, r;
    reg [8*256-1:0] header;

    initial begin
        fh = $fopen("weibull_vectors.txt","r");
        r = $fgets(header, fh);
        for (i=0;i<N;i=i+1)
            r = $fscanf(fh, "%d %d %d\n", v_c2[i], v_pfa[i], v_delta[i]);
        $fclose(fh);
        $display("Loaded %0d vectors.", N);
    end

    integer idx;
    integer errors, checked, chk_idx, diff;
    initial begin errors=0; checked=0; chk_idx = 0; end

    initial begin
        rstn = 0; molc_valid = 0;
        repeat(5) @(posedge clk);
        rstn = 1;
        @(posedge clk);
        for (idx = 0; idx < N; idx = idx + 1) begin
            molc_valid <= 1'b1;
            c2_code    <= v_c2[idx][25:0];
            pfa_sel    <= v_pfa[idx][1:0];
            @(posedge clk);
        end
        molc_valid <= 1'b0;
        repeat (50) @(posedge clk);
        $display("---- weibull_backend_tb: %0d checked, %0d errors ----", checked, errors);
        if (errors == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end

    always @(posedge clk) begin
        if (valid_out) begin
            if (chk_idx < N) begin
                diff = delta_out - v_delta[chk_idx];
                if (diff < 0) diff = -diff;
                if (diff > 1) begin
                    $display("MISMATCH idx=%0d c2=%0d pfa=%0d got=%0d expected=%0d",
                        chk_idx, v_c2[chk_idx], v_pfa[chk_idx], delta_out, v_delta[chk_idx]);
                    errors = errors + 1;
                end else begin
                    checked = checked + 1;
                end
                chk_idx = chk_idx + 1;
            end
        end
    end
endmodule
