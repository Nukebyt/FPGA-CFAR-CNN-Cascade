project_open prescreen_probe
create_timing_netlist
read_sdc
update_timing_netlist
report_timing -setup -npaths 12 -detail summary -panel_name "worst setup" -file worst_paths.txt
report_timing -setup -npaths 3 -detail full_path -file worst_full.txt
