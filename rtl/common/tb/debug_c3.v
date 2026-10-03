`timescale 1ns/1ps
module debug_c3;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rstn = 0;
    reg valid = 0;
    reg signed [22:0] s1w=0, s1g=0;
    reg signed [32:0] s2w=0, s2g=0;
    reg signed [33:0] s3w=0, s3g=0;

    wire mv;
    wire signed [15:0] c1c;
    wire [25:0] c2c;
    wire signed [27:0] c3c;

    molc_estimator3 dut(.clk(clk), .rstn(rstn), .sums_valid(valid),
        .sum1_window(s1w), .sum1_guard(s1g),
        .sum2_window(s2w), .sum2_guard(s2g),
        .sum3_window(s3w), .sum3_guard(s3g),
        .molc_valid(mv), .c1_code(c1c), .c2_code(c2c), .c3_code(c3c));

    initial begin
        rstn=0; valid=0;
        repeat(5) @(posedge clk);
        rstn=1;
        @(posedge clk);
        // Feed one representative window: modest positive sums
        s1w = 100; s1g=20; s2w=200; s2g=40; s3w=50; s3g=10; valid=1;
        @(posedge clk);
        valid=0;
        repeat(15) begin
            @(posedge clk);
            $display("t=%0t m1_2=%0d(v%b) s2n_2=%0d s3n_2=%0d | m1sq_raw_3=%0d(v%b) | m1sq_4=%0d m1s2_raw_4=%0d m1sq_raw_4=%0d(v%b) | m2_5=%0d m1s2_5=%0d m1cu_raw_5=%0d(v%b) | m2_6=%0d m3_6=%0d(v%b) | c2raw_7=%0d c3raw_7=%0d(v%b) | c2_8=%0d c3_8=%0d(v%b) | mv=%b c1=%0d c2=%0d c3=%0d",
                $time, dut.m1_2, dut.v2, dut.s2n_2, dut.s3n_2,
                dut.m1sq_raw_3, dut.v3,
                dut.m1sq_4, dut.m1s2_raw_4, dut.m1sq_raw_4, dut.v4,
                dut.m2_5, dut.m1s2_5, dut.m1cu_raw_5, dut.v5,
                dut.m2_6, dut.m3_6, dut.v6,
                dut.c2raw_7, dut.c3raw_7, dut.v7,
                dut.c2_8, dut.c3_8, dut.v8,
                mv, c1c, c2c, c3c);
        end
        $finish;
    end
endmodule
