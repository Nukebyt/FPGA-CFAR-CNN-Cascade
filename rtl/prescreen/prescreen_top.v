// prescreen_top.v -- pass 2 of the cascade: pooled-domain Weibull prescreen = prescreen_core + peak5.
// start (pulse) -> events (ev_valid with ev_j / ev_i on the pooled grid, ev_cd = A of the event, ev_pl = {S1, num>>12, P}) -> done.
`timescale 1ns/1ps
module prescreen_top #(
    parameter integer WP = 400, parameter integer HP = 400, parameter integer SLI = 25, parameter integer GUARD = 17,
    parameter integer AW = 18,
    parameter integer KC0 = 8381, parameter integer KC1 = 11060, parameter integer KC2 = 15733, parameter integer KC3 = 19546,
    parameter integer NUM_LO = 18371009, parameter integer NUM_HI = 1837100899
) (
    input  wire          clk, rstn, start,
    input  wire [1:0]    pfa_sel,
    input  wire [16:0]   g_th,
    output wire [AW-1:0] ps_raddr,
    input  wire [7:0]    ps_rdata,
    output wire          busy,
    output wire          ev_valid,
    output wire [9:0]    ev_j, ev_i,
    output wire [16:0]   ev_cd,
    output wire [43:0]   ev_pl,
    output wire          done
);
    wire o_valid, o_last; wire [16:0] o_cd; wire [43:0] o_pl;
    prescreen_core #(.WP(WP), .HP(HP), .SLI(SLI), .GUARD(GUARD), .AW(AW), .KC0(KC0), .KC1(KC1), .KC2(KC2), .KC3(KC3),
                     .NUM_LO(NUM_LO), .NUM_HI(NUM_HI)) core (
        .clk(clk), .rstn(rstn), .start(start), .pfa_sel(pfa_sel), .g_th(g_th), .ps_raddr(ps_raddr), .ps_rdata(ps_rdata),
        .busy(busy), .o_valid(o_valid), .o_cd(o_cd), .o_pl(o_pl), .o_last(o_last));
    peak5 #(.WP(WP), .HP(HP), .CW(17), .PW(44)) pk (
        .clk(clk), .rstn(rstn), .start(start), .in_valid(o_valid), .in_cd(o_cd), .in_pl(o_pl), .in_last(o_last),
        .ev_valid(ev_valid), .ev_j(ev_j), .ev_i(ev_i), .ev_cd(ev_cd), .ev_pl(ev_pl), .done(done));
endmodule
