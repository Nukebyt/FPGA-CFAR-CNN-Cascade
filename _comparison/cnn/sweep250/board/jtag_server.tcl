# jtag_server.tcl -- run with:  system-console --script=jtag_server.tcl [port]
# A tiny TCP command server around the JTAG-to-Avalon master of cascade_jtag_top (python: board.py talks to it).
# Protocol: one Tcl command per line in, one line out:  "<rc> <result>"   (newlines in results are escaped as \n).
set port 5555
if {$argc > 0} { set port [lindex $argv 0] }
set paths [get_service_paths master]
puts "master service paths: $paths"
set m [lindex $paths 0]
open_service master $m
puts "opened $m"

proc handle {chan} {
    if {[gets $chan line] < 0} { close $chan; return }
    set rc [catch {uplevel #0 $line} res]
    puts $chan "$rc [string map [list \n {\n}] $res]"
    flush $chan
}
proc accept {chan addr port} {
    fconfigure $chan -buffering line -translation lf
    fileevent $chan readable [list handle $chan]
}
socket -server accept $port
puts "jtag_server listening on $port"
vwait forever
