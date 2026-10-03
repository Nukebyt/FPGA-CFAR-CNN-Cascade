create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_pll_clocks
derive_clock_uncertainty
set_false_path -from [get_ports {KEY[*]}]
set_false_path -to [get_ports {LEDR[*]}]
set_false_path -to [get_ports {HEX*}]
# stream (CLOCK_50) and CNN (PLL 100 MHz) domains exchange only synchronised handshake levels and quasi-static data
set_clock_groups -asynchronous -group {CLOCK_50} -group [get_clocks {pll|u_pll|*|outclk*}]
# USB-Blaster JTAG clock
create_clock -name altera_reserved_tck -period 100.000 [get_ports -nowarn altera_reserved_tck]
set_clock_groups -asynchronous -group {altera_reserved_tck} -group {CLOCK_50} -group [get_clocks {pll|u_pll|*|outclk*}]
