# Platform Designer system for the cascade JTAG bring-up: a 32-bit JTAG-to-Avalon master (System Console) whose
# Avalon-MM master interface is exported to the top level, where it drives cascade_jtag_core's slave port.
package require -exact qsys 16.1
create_system jtag_sys
set_project_property DEVICE_FAMILY "Cyclone V"
set_project_property DEVICE 5CSXFC6D6F31C6
add_instance clk_0 clock_source
set_instance_parameter_value clk_0 clockFrequency 50000000
set_instance_parameter_value clk_0 clockFrequencyKnown true
add_instance jtag_master altera_jtag_avalon_master
set_instance_parameter_value jtag_master USE_PLI 0
set_instance_parameter_value jtag_master PLI_PORT 50000
set_instance_parameter_value jtag_master FAST_VER 0
set_instance_parameter_value jtag_master FIFO_DEPTHS 2
add_connection clk_0.clk jtag_master.clk clock
add_connection clk_0.clk_reset jtag_master.clk_reset reset
add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_0.clk_in_reset
add_interface avm avalon master
set_interface_property avm EXPORT_OF jtag_master.master
save_system jtag_sys.qsys
