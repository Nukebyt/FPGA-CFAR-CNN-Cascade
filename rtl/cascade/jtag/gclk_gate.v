// gclk_gate.v -- glitch-free clock gate (Cyclone V clock-control block with registered enable).
//
// The Weibull front end needs a gap-free pixel stream (KNOWN_ISSUE_GAP_INTOLERANCE.md) but a JTAG / HPS host delivers
// pixels slowly and irregularly. Instead of buffering a whole 800x800 frame (5.1 Mbit, more than the free M10K), the
// stream-domain clock is simply STOPPED while no pixel is available: every flop of the Weibull / gate / pool-store
// logic freezes, so the core sees an ideal gap-free stream no matter how slowly pixels arrive.
//
// CE semantics (both in simulation and in the clock-control block, ena_register_mode = "falling edge"):
//   `en` is changed by flops on the rising edge of `clk`;   the NEXT rising edge of `gclk` exists iff `en` was 1.
// Define SIM_GATE for simulation (behavioural: enable latched while clk is low).
`timescale 1ns/1ps
module gclk_gate (
    input  wire clk,
    input  wire en,
    output wire gclk
);
`ifdef SIM_GATE
    reg en_l = 1'b1;
    always @(clk or en) if (!clk) en_l = en;       // transparent latch while clk is low: the enable set at the previous rising edge is held across the next one
    assign gclk = clk & en_l;
`else
    altclkctrl #(.clock_type("Global Clock"), .ena_register_mode("falling edge"), .number_of_clocks(1), .width_clkselect(1))
        u_gate (.inclk(clk), .ena(en), .outclk(gclk));
`endif
endmodule
