`timescale 1ns/1ps
module front_end3_probe_tb;
    localparam integer SLI=17, GUARD=13, IMG_W=48, IMG_H=40;
    reg clk=0; always #5 clk=~clk;
    reg rstn=0, pixel_in_valid=0;
    reg [7:0] pixel_in=0;
    wire molc_valid;
    wire signed [15:0] c1_code;
    wire [25:0] c2_code;
    wire signed [27:0] c3_code;
    wire signed [15:0] x_code;

    front_end3 #(.SLI(SLI), .GUARD(GUARD), .IMG_WIDTH(IMG_W), .IMG_HEIGHT(IMG_H),
        .LOG_AMP_HEX("../../../lut/shared/log_amp_lut.hex")) dut (
        .clk(clk), .rstn(rstn), .pixel_in_valid(pixel_in_valid), .pixel_in(pixel_in),
        .molc_valid(molc_valid), .c1_code(c1_code), .c2_code(c2_code), .c3_code(c3_code), .x_code(x_code));

    reg [7:0] pixmem [0:IMG_W*IMG_H-1];
    integer i, fh, r3, pixidx;
    integer mv_count;

    initial begin
        fh=$fopen("fe3_pixels.txt","r");
        for (i=0;i<IMG_W*IMG_H;i=i+1) r3=$fscanf(fh,"%d\n",pixmem[i]);
        $fclose(fh);
    end

    initial begin
        rstn=0; pixel_in_valid=0; mv_count=0; pixidx=0;
        repeat(5) @(posedge clk);
        rstn=1;
        @(posedge clk);
        for (pixidx=0; pixidx<IMG_W*IMG_H; pixidx=pixidx+1) begin
            pixel_in_valid<=1'b1;
            pixel_in<=pixmem[pixidx];
            @(posedge clk);
        end
        pixel_in_valid<=1'b0;
        repeat(50) @(posedge clk);
        $finish;
    end

    always @(posedge clk) begin
        if (molc_valid) begin
            mv_count=mv_count+1;
            if (mv_count<=5)
                $display("t=%0t mv#%0d c1=%0d c2=%0d c3=%0d | dut.wv_x=%b dut.s1v=%b dut.xc1v=%b",
                    $time, mv_count, c1_code, c2_code, c3_code, dut.wv_x, dut.s1v, dut.xc1v);
            $display("   me3: m1_2=%0d s2n_2=%0d s3n_2=%0d(v2=%b) | m1sq_raw_3=%0d(v3=%b) | m1s2_raw_4=%0d m1sq_raw_4=%0d(v4=%b) | m2_5=%0d m1s2_5=%0d m1cu_raw_5=%0d(v5=%b) | m2_6=%0d m3_6=%0d(v6=%b) | c3raw_7=%0d(v7=%b) | c3_8=%0d(v8=%b)",
                dut.me3.m1_2, dut.me3.s2n_2, dut.me3.s3n_2, dut.me3.v2,
                dut.me3.m1sq_raw_3, dut.me3.v3,
                dut.me3.m1s2_raw_4, dut.me3.m1sq_raw_4, dut.me3.v4,
                dut.me3.m2_5, dut.me3.m1s2_5, dut.me3.m1cu_raw_5, dut.me3.v5,
                dut.me3.m2_6, dut.me3.m3_6, dut.me3.v6,
                dut.me3.c3raw_7, dut.me3.v7,
                dut.me3.c3_8, dut.me3.v8);
        end
    end
endmodule
