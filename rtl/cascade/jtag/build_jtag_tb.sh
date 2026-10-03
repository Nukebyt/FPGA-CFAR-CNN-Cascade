#!/bin/bash
# usage: build_jtag_tb.sh <gen_dir under rtl/cnn> <theta1> [two=0]   -- 128x128 crop through cascade_jtag_core
cd /f/Projects/CFAR/rtl/cascade
L=/f/Projects/CFAR
mkdir -p tb/out_jtag
iverilog -g2012 -DSIM_PLL -DSIM_GATE -I ../cnn/$1 -DQ4V=1 -DTHETA1=$2 -DTWO=${3:-0} -DIMGW=128 -DIMGH=128 -DIMGHEX=\"tb/img128.hex\" -DWHEX=\"../cnn/$1/cnn_w.hex\" -DPQHEX=\"../cnn/$1/cnn_pq.hex\" -DOUTDIR=\"tb/out_jtag\" \
 -o sim_jtag.vvp jtag/tb/cascade_jtag_tb.v jtag/cascade_jtag_core.v jtag/gclk_gate.v cascade_top_2clk.v cnn_pll.v weibull_front_cascade.v trigger_gate.v pool_store_writer.v pooled_store_dc.v event_ram_dc.v patch_fetch.v ../cnn/cnn_core.v ../cnn/cnn_core_q4.v \
 $L/CFAR_Weibull/rtl/weibull_backend_new.v $L/CFAR_Weibull/rtl/weibull_delta_rom.v $L/rtl/common/front_end3.v $L/rtl/common/power_expand.v $L/rtl/common/box_sum_inc.v $L/rtl/common/molc_estimator3.v $L/rtl/common/log_amp_rom.v $L/rtl/common/line_buffer.v $L/rtl/common/row_delay_mem.v $L/rtl/common/pipe_delay.v $L/rtl/common/linear_addr.v $L/rtl/common/threshold_compare3.v \
 && vvp sim_jtag.vvp
