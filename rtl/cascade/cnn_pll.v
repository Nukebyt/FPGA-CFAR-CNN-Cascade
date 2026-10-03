// cnn_pll.v -- 50 MHz -> 100 MHz clock for the CNN domain (Cyclone V altera_pll, direct instantiation).
// Define SIM_PLL for simulation: a behavioural 100 MHz clock with arbitrary phase to the 50 MHz domain.
`timescale 1ns/1ps
module cnn_pll #(parameter OUT_MHZ = "100.000000 MHz") (
    input  wire refclk,
    input  wire rst,
    output wire outclk,
    output wire locked
);
`ifdef SIM_PLL
    reg c = 1'b0;
    initial #3 c = 1'b1;                 // not aligned to refclk edges
    always #5 c = ~c;
    assign outclk = c;
    assign locked = 1'b1;
`else
    altera_pll #(
        .fractional_vco_multiplier("false"),
        .reference_clock_frequency("50.0 MHz"),
        .operation_mode("direct"),
        .number_of_clocks(1),
        .output_clock_frequency0(OUT_MHZ),
        .phase_shift0("0 ps"),
        .duty_cycle0(50),
        .pll_type("General"),
        .pll_subtype("General")
    ) u_pll (
        .rst(rst), .outclk(outclk), .locked(locked), .fboutclk(), .fbclk(1'b0), .refclk(refclk)
    );
`endif
endmodule
