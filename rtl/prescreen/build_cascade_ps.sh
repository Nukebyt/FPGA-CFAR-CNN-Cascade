#!/bin/bash
# usage: build_cascade_ps.sh <genq_dir> <imgw> <imgh> <theta_neg> <imghex> <outdir> <simname> <sli> <guard> <pfa> <gth> "<KC0 KC1 KC2 KC3 NUM_LO NUM_HI>" [q4=1] [gaps=0]
cd /f/Projects/CFAR/rtl/prescreen
R=/f/Projects/CFAR/rtl
read KC0 KC1 KC2 KC3 NUMLO NUMHI <<< "${12}"
iverilog -g2012 $EXTRA -DSIM_PLL -I $R/cnn/$1 -DQ4V=${13:-1} -DGAPS=${14:-0} -DIMGW=$2 -DIMGH=$3 -DTHETA=$4 -DEVAW=13 -DSLI=$8 -DGUARD=$9 -DPFA=${10} -DGTH=${11} \
 -DKC0=$KC0 -DKC1=$KC1 -DKC2=$KC2 -DKC3=$KC3 -DNUM_LO=$NUMLO -DNUM_HI=$NUMHI -DQROMHEX=\"$R/cascade/qrom.hex\" \
 -DWHEX=\"$R/cnn/$1/cnn_w.hex\" -DPQHEX=\"$R/cnn/$1/cnn_pq.hex\" -DIMGHEX=\"$5\" -DOUTDIR=\"$6\" \
 -o $7 tb/cascade_ps_tb.v cascade_ps_2clk.v prescreen_top.v prescreen_core.v peak5.v $R/cascade/cnn_pll.v $R/cascade/pool_store_writer.v $R/cascade/pooled_store_dc.v $R/cascade/event_ram_dc.v $R/cascade/patch_fetch.v $R/cnn/cnn_core.v $R/cnn/cnn_core_q4.v
