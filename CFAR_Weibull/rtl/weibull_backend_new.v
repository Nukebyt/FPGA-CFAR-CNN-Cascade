// weibull_backend_new.v
// Weibull on the NEW shared front end / unified back-end architecture
// (FINDINGS F1/F10: identical structure to lognormal_backend.v, differing
// only in the ROM contents and address constants -- Weibull and Lognormal
// are the same decision rule up to one Pfa-dependent constant). Named
// "_new" to distinguish from the existing hardware-validated Weibull_CFAR
// project's own RTL (window_sum.v/molc_estimator.v/... at the OLD Q2.13,
// 2-moment precision), which this does not replace or touch.
module weibull_backend_new #(
    parameter LUT_ROOT = "lut"
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        molc_valid,
    input  wire [25:0] c2_code,
    input  wire [1:0]  pfa_sel,

    output wire        valid_out,
    output wire signed [15:0] delta_out,
    output wire        addr_sat_low,
    output wire        addr_sat_high
);
    localparam signed [47:0] WB_SCALE_Q  = 48'sd94923257; // frac16
    localparam signed [95:0] WB_OFFSET_Q = 96'sd0;        // vmin=0 exactly

    wire addr_valid; wire [11:0] c2_addr;
    wire sl, sh;
    linear_addr #(.IN_WIDTH(27), .X_FRAC(24), .SCALE_FRAC(16), .ADDR_BITS(12),
        .SCALE_Q(WB_SCALE_Q), .OFFSET_Q(WB_OFFSET_Q)) u_addr (
        .clk(clk), .valid_in(molc_valid), .x($signed({1'b0,c2_code})),
        .valid_out(addr_valid), .addr(c2_addr), .sat_low(sl), .sat_high(sh));

    wire [1:0] pfa_sel_d1;
    pipe_delay #(.DATA_WIDTH(2), .DEPTH(1)) d_pfa (
        .clk(clk), .rstn(rstn), .in_valid(1'b1), .in_data(pfa_sel),
        .out_valid(), .out_data(pfa_sel_d1));

    wire signed [15:0] delta_code;
    weibull_delta_rom #(.HEX_FILE({LUT_ROOT,"/weibull/weibull_delta_lut.hex"})) u_delta (
        .clk(clk), .pfa_sel(pfa_sel_d1), .c2_addr(c2_addr), .dout(delta_code));

    reg valid_final, satl_f, sath_f;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin valid_final<=0; satl_f<=0; sath_f<=0; end
        else begin valid_final <= addr_valid; satl_f <= sl; sath_f <= sh; end
    end

    assign valid_out = valid_final;
    assign delta_out = delta_code;
    assign addr_sat_low  = satl_f;
    assign addr_sat_high = sath_f;
endmodule
