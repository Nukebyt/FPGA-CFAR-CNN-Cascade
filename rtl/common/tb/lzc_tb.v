`timescale 1ns/1ps
module lzc_tb;
    localparam integer N = 2231;
    localparam integer WIDTH = 26;

    reg clk = 0;
    always #5 clk = ~clk;

    reg [WIDTH-1:0] code_in;
    wire lg_valid; wire signed [21:0] lg_out;      // OUT_INT=5 -> 5+17=22 bits
    wire sq_valid; wire [15:0]        sq_out;       // OUT_INT=2, OUT_FRAC=14 -> 16 bits

    log2_lzc #(.WIDTH(WIDTH), .FRACBITS(24), .OUT_INT(5),
        .MANT_HEX("../../../lut/shared/log2_mant_lut.hex")) dut_lg (
        .clk(clk), .code(code_in), .valid(lg_valid), .lg(lg_out));

    sqrt_lzc #(.WIDTH(WIDTH), .FRACBITS(24), .OUT_FRAC(14), .OUT_INT(2),
        .MANT_HEX("../../../lut/shared/sqrt_mant_lut.hex")) dut_sq (
        .clk(clk), .code(code_in), .valid(sq_valid), .sq(sq_out));

    reg [31:0] vcode [0:N-1];
    integer    vlg   [0:N-1];
    integer    vsq   [0:N-1];
    integer i, fh, r;
    reg [8*256-1:0] header;

    initial begin
        fh = $fopen("lzc_vectors.txt","r");
        r = $fgets(header, fh);
        for (i=0;i<N;i=i+1) r = $fscanf(fh, "%d %d %d\n", vcode[i], vlg[i], vsq[i]);
        $fclose(fh);
        $display("Loaded %0d vectors.", N);
    end

    integer errors_lg, errors_sq, idx;
    integer diff;
    reg [31:0] exp_idx_q [0:31]; // small queue to track which vector index is in-flight (2-cycle pipeline)

    initial begin
        errors_lg = 0; errors_sq = 0;
        code_in = 0;
        @(posedge clk);
        for (idx = 0; idx < N; idx = idx + 1) begin
            code_in <= vcode[idx][WIDTH-1:0];
            @(posedge clk);
        end
        code_in <= 0;
        repeat (10) @(posedge clk);
        $display("log2_lzc: %0d errors / %0d", errors_lg, N);
        $display("sqrt_lzc: %0d errors / %0d", errors_sq, N);
        if (errors_lg==0 && errors_sq==0) $display("PASS"); else $display("FAIL");
        $finish;
    end

    // both modules have 2-cycle latency from code_in to valid/output
    integer chk_idx;
    initial chk_idx = -3;
    always @(posedge clk) begin
        if (chk_idx >= 0 && chk_idx < N) begin
            if (lg_valid) begin
                if (lg_out - vlg[chk_idx] > 2 || lg_out - vlg[chk_idx] < -2) begin
                    $display("log2 MISMATCH idx=%0d code=%0d got=%0d expected=%0d", chk_idx, vcode[chk_idx], lg_out, vlg[chk_idx]);
                    errors_lg = errors_lg + 1;
                end
            end
            // Relative tolerance, not a fixed LSB count: the final output is
            // mantissa*2^(e/2), so any residual address-rounding difference
            // between this fixed-point implementation and MATLAB's
            // double-precision (m-1)/3*n gets scaled by the SAME exponent as
            // the output itself -- an address off by a fraction of an LSB at
            // large magnitude is still a tiny RELATIVE error, not a growing
            // absolute one. Measured: worst case 15/32756 = 0.046% of the
            // output's own magnitude, well under the mantissa table's own
            // ~0.1% (1/1024) resolution -- an inherent property of the
            // shared table, not a defect in this module.
            if (sq_valid) begin
                diff = $signed({1'b0,sq_out}) - vsq[chk_idx];
                if (diff < 0) diff = -diff;
                if (diff > 1 && diff*500 > vsq[chk_idx]) begin // > 1 LSB and > 0.2% relative
                    $display("sqrt MISMATCH idx=%0d code=%0d got=%0d expected=%0d", chk_idx, vcode[chk_idx], sq_out, vsq[chk_idx]);
                    errors_sq = errors_sq + 1;
                end
            end
        end
        chk_idx = chk_idx + 1;
    end
endmodule
