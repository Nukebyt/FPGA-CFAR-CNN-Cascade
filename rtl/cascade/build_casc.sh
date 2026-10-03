#!/bin/bash
# usage: build_casc.sh <gen_dir> <imgw> <imgh> <tau_q14> <theta_neg> <imghex> <outdir> <simname> [q4=0|1]   (q4=1 needs a genq_* dir)
cd /f/Projects/CFAR/rtl/cascade
L=/f/Projects/CFAR
iverilog -g2012 -I ../cnn/$1 -DQ4V=${9:-0} -DIMGW=$2 -DIMGH=$3 -DTAU=$4 -DTHETA=$5 -DLUTROOT=\"F:/Projects/CFAR/lut\" -DQROMHEX=\"qrom.hex\" \
 -DWHEX=\"../cnn/$1/cnn_w.hex\" -DPQHEX=\"../cnn/$1/cnn_pq.hex\" -DIMGHEX=\"$6\" -DOUTDIR=\"$7\" \
 -o $8 tb/cascade_tb.v cascade_top.v weibull_front_cascade.v trigger_gate.v pool_store_writer.v pooled_store.v patch_fetch.v event_fifo.v ../cnn/cnn_core.v ../cnn/cnn_core_q4.v \
 $L/CFAR_Weibull/rtl/weibull_backend_new.v $L/CFAR_Weibull/rtl/weibull_delta_rom.v $L/rtl/common/front_end3.v $L/rtl/common/power_expand.v $L/rtl/common/box_sum_inc.v $L/rtl/common/molc_estimator3.v $L/rtl/common/log_amp_rom.v $L/rtl/common/line_buffer.v $L/rtl/common/row_delay_mem.v $L/rtl/common/pipe_delay.v $L/rtl/common/linear_addr.v $L/rtl/common/threshold_compare3.v
