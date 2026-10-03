#!/bin/bash
# usage: build_cascade_ctx.sh <gen_dir_abs> <imgw> <imgh> <theta_neg> <imghex> <outdir> <simname> <pfa> <gth> [gaps=0]   (config A: SLI 25 / GUARD 17)
cd /f/Projects/CFAR/rtl/ctx
R=/f/Projects/CFAR/rtl
iverilog -g2012 $EXTRA -DSIM_PLL -I $1 -DGAPS=${10:-0} -DIMGW=$2 -DIMGH=$3 -DTHETA=$4 -DEVAW=12 -DSLI=25 -DGUARD=17 -DPFA=$8 -DGTH=$9 \
 -DKC0=8381 -DKC1=11060 -DKC2=15733 -DKC3=19546 -DNUM_LO=18371009 -DNUM_HI=1837100899 -DQROMHEX=\"$R/cascade/qrom.hex\" -DLNHEX=\"$R/ctx/ln_rom.hex\" \
 -DWHEX=\"$1/cnn_w.hex\" -DPQHEX=\"$1/cnn_pq.hex\" -DIMGHEX=\"$5\" -DOUTDIR=\"$6\" \
 -o $7 tb/cascade_ctx_tb.v cascade_ctx_2clk.v side_unit.v isqrt_pipe.v img_stats.v img_codes.v ln_rom.v ctx_fetch.v $R/prescreen/prescreen_top.v $R/prescreen/prescreen_core.v $R/prescreen/peak5.v \
 $R/cascade/cnn_pll.v $R/cascade/pool_store_writer.v $R/cascade/pooled_store_dc.v $R/cascade/event_ram_dc.v $R/cascade/patch_fetch.v $R/cnn/cnn_core_ctx.v
