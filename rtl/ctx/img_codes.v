// img_codes.v -- image side features f4..f7 from the accumulated statistics (golden model: side_fx.image_codes), clk_cnn domain, once per frame.
//   R = round(2^40 / NPIX)
//   f4 = (sumQ*R + 2^39) >> 40     f5 = isqrt(max(0, ((sumQ2*R + 2^39) >> 40) - f4^2))
//   f6 = min(255, (cntB*R + 2^28) >> 29)     f7 = min(255, (cntD*255*R + 2^39) >> 40)
// Shift-add multiplies, one per ~22 clocks; total about 100 clocks per frame.
`timescale 1ns/1ps
module img_codes #(parameter integer NPIX = 640000) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        start,
    input  wire [27:0] sumq,
    input  wire [35:0] sumq2,
    input  wire [19:0] cnt_b,
    input  wire [19:0] cnt_d,
    output reg  [7:0]  f4, f5, f6, f7,
    output reg         done
);
    localparam [63:0] RR = ((64'd1 << 40) + NPIX / 2) / NPIX;
    localparam [31:0] R = RR[31:0];
    reg [3:0]  st;
    reg [71:0] a, prod;
    reg [31:0] b; reg [5:0] cnt;
    reg [15:0] e2;
    reg        sq_v; reg [15:0] sq_x; wire sq_o; wire [7:0] sq_r;
    isqrt_pipe sq (.clk(clk), .in_valid(sq_v), .x(sq_x), .out_valid(sq_o), .root(sq_r));
    wire [63:0] var_w = e2 - f4 * f4;
    always @(posedge clk) begin
        if (!rstn) begin st <= 4'd0; done <= 1'b0; sq_v <= 1'b0; f4 <= 8'd0; f5 <= 8'd0; f6 <= 8'd0; f7 <= 8'd0; end
        else begin
            done <= 1'b0; sq_v <= 1'b0;
            case (st)
                4'd0: if (start) begin a <= {44'd0, sumq}; b <= R; prod <= 72'd0; cnt <= 6'd32; st <= 4'd1; end
                4'd1: begin                                          // shift-add multiply a*b (b = constant R, 21 bits)
                    if (cnt != 0) begin
                        if (b[0]) prod <= prod + a;
                        a <= a << 1; b <= b >> 1; cnt <= cnt - 1'b1;
                    end else st <= 4'd2;
                end
                4'd2: begin f4 <= (prod + (72'd1 << 39)) >> 40; a <= {36'd0, sumq2}; b <= R; prod <= 72'd0; cnt <= 6'd32; st <= 4'd3; end
                4'd3: begin
                    if (cnt != 0) begin if (b[0]) prod <= prod + a; a <= a << 1; b <= b >> 1; cnt <= cnt - 1'b1; end
                    else st <= 4'd4;
                end
                4'd4: begin e2 <= ((prod + (72'd1 << 39)) >> 40); st <= 4'd5; end
                4'd5: begin                                          // var = e2 - f4^2, clamp, root
                    sq_x <= (e2 < f4 * f4) ? 16'd0 : (var_w > 64'd65535 ? 16'hFFFF : var_w[15:0]); sq_v <= 1'b1; st <= 4'd6; 
                end
                4'd6: begin
                    if (sq_o) begin f5 <= sq_r; a <= {52'd0, cnt_b}; b <= R; prod <= 72'd0; cnt <= 6'd32; st <= 4'd7; end
                end
                4'd7: begin
                    if (cnt != 0) begin if (b[0]) prod <= prod + a; a <= a << 1; b <= b >> 1; cnt <= cnt - 1'b1; end
                    else st <= 4'd8;
                end
                4'd8: begin f6 <= (((prod + (72'd1 << 28)) >> 29) > 72'd255) ? 8'd255 : ((prod + (72'd1 << 28)) >> 29);
                            a <= {52'd0, cnt_d} * 8'd255; b <= R; prod <= 72'd0; cnt <= 6'd32; st <= 4'd9; end
                4'd9: begin
                    if (cnt != 0) begin if (b[0]) prod <= prod + a; a <= a << 1; b <= b >> 1; cnt <= cnt - 1'b1; end
                    else st <= 4'd10;
                end
                4'd10: begin f7 <= (((prod + (72'd1 << 39)) >> 40) > 72'd255) ? 8'd255 : ((prod + (72'd1 << 39)) >> 40); done <= 1'b1; st <= 4'd0; end
                default: st <= 4'd0;
            endcase
        end
    end
endmodule
