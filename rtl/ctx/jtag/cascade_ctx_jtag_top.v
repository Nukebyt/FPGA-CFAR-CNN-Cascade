// cascade_ctx_jtag_top.v -- DE10-Standard top for the JTAG-fed cascade with the pooled-domain prescreen (Paper 2).  PC (System Console) -> JTAG master (Qsys)
// -> cascade_jtag_core.  Full 800x800 frames, any Pfa plane / gate / CNN threshold, events read back over the same JTAG.
//   LEDR[0] ready   [1] loading   [2] frame done   [3] heartbeat   [4] event overflow   [5] PLL locked   [9:6] accepted (sat 15)
//   HEX5..3 candidate events (hex)   HEX2..0 accepted events (hex)        KEY[0] reset
`timescale 1ns/1ps
module cascade_ctx_jtag_top #(
    parameter CNN_W_HEX  = "F:/Projects/CFAR/rtl/cnn/ctx_fx/cnn_w.hex",
    parameter CNN_PQ_HEX = "F:/Projects/CFAR/rtl/cnn/ctx_fx/cnn_pq.hex"
) (
    input  wire        CLOCK_50,
    input  wire [1:0]  KEY,
    output wire [9:0]  LEDR,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5
);
    wire rstn = KEY[0];
    wire clk_cnn, pll_locked;
    cnn_pll pll (.refclk(CLOCK_50), .rst(1'b0), .outclk(clk_cnn), .locked(pll_locked));

    wire [31:0] avm_address, avm_readdata, avm_writedata; wire [3:0] avm_byteenable;
    wire avm_read, avm_write, avm_waitrequest, avm_readdatavalid;
    jtag_sys js (.clk_clk(CLOCK_50), .reset_reset_n(rstn),
        .avm_address(avm_address), .avm_readdata(avm_readdata), .avm_read(avm_read), .avm_write(avm_write),
        .avm_writedata(avm_writedata), .avm_waitrequest(avm_waitrequest), .avm_readdatavalid(avm_readdatavalid),
        .avm_byteenable(avm_byteenable));

    wire st_ready, st_loading, st_done, st_ovf; wire [15:0] n_ev, n_acc;
    cascade_ctx_jtag_core #(.IMG_W(800), .IMG_H(800),
                        .QROM_HEX("F:/Projects/CFAR/rtl/cascade/qrom.hex"),
                        .CNN_W_HEX(CNN_W_HEX), .CNN_PQ_HEX(CNN_PQ_HEX), .LN_HEX("F:/Projects/CFAR/rtl/ctx/ln_rom.hex")) core (
        .clk(CLOCK_50), .clk_cnn(clk_cnn), .rstn_in(rstn), .pll_locked(pll_locked),
        .avs_address(avm_address[17:2]), .avs_win(|avm_address[31:18]), .avs_write(avm_write), .avs_writedata(avm_writedata), .avs_read(avm_read),
        .avs_readdata(avm_readdata), .avs_readdatavalid(avm_readdatavalid), .avs_waitrequest(avm_waitrequest),
        .st_ready(st_ready), .st_loading(st_loading), .st_done(st_done), .st_ovf(st_ovf),
        .st_n_events(n_ev), .st_n_accepted(n_acc));

    reg [25:0] hb;
    always @(posedge CLOCK_50 or negedge rstn) if (!rstn) hb <= 26'd0; else hb <= hb + 1'b1;
    wire [3:0] acc_sat = (n_acc > 15) ? 4'd15 : n_acc[3:0];
    assign LEDR = {acc_sat, pll_locked, st_ovf, hb[25], st_done, st_loading, st_ready};

    function [6:0] seg;
        input [3:0] v;
        case (v)
            4'h0: seg = 7'b1000000; 4'h1: seg = 7'b1111001; 4'h2: seg = 7'b0100100; 4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001; 4'h5: seg = 7'b0010010; 4'h6: seg = 7'b0000010; 4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000; 4'h9: seg = 7'b0010000; 4'hA: seg = 7'b0001000; 4'hB: seg = 7'b0000011;
            4'hC: seg = 7'b1000110; 4'hD: seg = 7'b0100001; 4'hE: seg = 7'b0000110; default: seg = 7'b0001110;
        endcase
    endfunction
    assign HEX0 = seg(n_acc[3:0]);  assign HEX1 = seg(n_acc[7:4]);  assign HEX2 = seg(n_acc[11:8]);
    assign HEX3 = seg(n_ev[3:0]);   assign HEX4 = seg(n_ev[7:4]);   assign HEX5 = seg(n_ev[11:8]);
endmodule
