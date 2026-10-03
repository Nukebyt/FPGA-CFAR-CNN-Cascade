#!/bin/bash
# usage: build_jtag_ps_tb.sh <genq_dir> <theta_neg> <imgw> <imgh> <imghex> <outdir> <pfa> <gth> [two=0]  -- crop through cascade_ps_jtag_core (register-level flow, gappy host)
cd /f/Projects/CFAR/rtl/prescreen
R=/f/Projects/CFAR/rtl
mkdir -p $6
iverilog -g2012 -DSIM_PLL -I $R/cnn/$1 -DQ4V=1 -DTHETA1=$2 -DTWO=${9:-0} -DPFA=$7 -DGTH=$8 -DIMGW=$3 -DIMGH=$4 -DIMGHEX=\"$5\" -DWHEX=\"$R/cnn/$1/cnn_w.hex\" -DPQHEX=\"$R/cnn/$1/cnn_pq.hex\" -DOUTDIR=\"$6\" \
 -o sim_jtag_ps.vvp jtag/tb/cascade_ps_jtag_tb.v jtag/cascade_ps_jtag_core.v cascade_ps_2clk.v prescreen_top.v prescreen_core.v peak5.v $R/cascade/cnn_pll.v $R/cascade/pool_store_writer.v $R/cascade/pooled_store_dc.v $R/cascade/event_ram_dc.v $R/cascade/patch_fetch.v $R/cnn/cnn_core.v $R/cnn/cnn_core_q4.v
