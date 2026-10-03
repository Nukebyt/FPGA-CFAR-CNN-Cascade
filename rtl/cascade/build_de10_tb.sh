#!/bin/bash
cd /f/Projects/CFAR/rtl/cascade
L=/f/Projects/CFAR
iverilog -g2012 -DSIM_PLL -I ../cnn/$1 -DQ4V=${3:-0} -DTHETA1=$2 -DIMGHEX=\"tb/img128.hex\" -DWHEX=\"../cnn/$1/cnn_w.hex\" -DPQHEX=\"../cnn/$1/cnn_pq.hex\" -DOUTDIR=\"tb/out_de10\" \
 -o sim_de10.vvp tb/cascade_de10_top_tb.v cascade_de10_top.v cascade_top_2clk.v cnn_pll.v weibull_front_cascade.v trigger_gate.v pool_store_writer.v pooled_store_dc.v event_ram_dc.v patch_fetch.v ../cnn/cnn_core.v ../cnn/cnn_core_q4.v \
 $L/CFAR_Weibull/rtl/weibull_backend_new.v $L/CFAR_Weibull/rtl/weibull_delta_rom.v $L/rtl/common/front_end3.v $L/rtl/common/power_expand.v $L/rtl/common/box_sum_inc.v $L/rtl/common/molc_estimator3.v $L/rtl/common/log_amp_rom.v $L/rtl/common/line_buffer.v $L/rtl/common/row_delay_mem.v $L/rtl/common/pipe_delay.v $L/rtl/common/linear_addr.v $L/rtl/common/threshold_compare3.v
