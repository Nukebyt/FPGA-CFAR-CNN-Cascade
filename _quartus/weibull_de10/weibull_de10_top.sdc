# gengamma_de10_top.sdc -- TimeQuest timing constraints for the board wrapper.
# The DE10-Standard's real onboard oscillator, on the real CLOCK_50 pin --
# unlike the resource-probe project's "clk" port, this one drives actual
# silicon, so its Fmax/slack numbers are the ones that matter for the
# physical bring-up decision.

create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty

# KEY[0]/KEY[1] are asynchronous board reset / (unused) inputs.
set_false_path -from [get_ports {KEY[0]}]
set_false_path -from [get_ports {KEY[1]}]

# SW[1:0] (pfa_sel) is a slow-changing human input, resynchronized by the
# wrapper's own registered pfa_sel_r -- exclude it from setup/hold, matching
# how a real switch/button input is always meant to be treated.
set_false_path -from [get_ports {SW[*]}]

# LEDR outputs are purely observational (human eyes, not another synchronous
# device) -- no meaningful timing requirement on them either.
set_false_path -to [get_ports {LEDR[*]}]
