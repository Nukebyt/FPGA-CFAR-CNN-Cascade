#!/bin/bash
# bit-exact regression of cnn_core_q4 against the exported golden vectors (all three hardware-faithful models)
cd /f/Projects/CFAR/rtl/cnn
for spec in "small 64" "deep 48" "xl 12"; do
  set -- $spec; m=$1; n=$2
  E=../../_comparison/Results/cnn_weights_hw/hw_$m
  iverilog -g2012 -I genq_hw_$m -DCORE=cnn_core_q4 -DWHEX=\"genq_hw_$m/cnn_w.hex\" -DPQHEX=\"genq_hw_$m/cnn_pq.hex\" -DGIN=\"$E/golden_in.hex\" -DGLG=\"$E/golden_logit_int.txt\" -DNV=$n -o sim_q4_$m.vvp cnn_core_q4.v tb/cnn_core_tb.v 2>&1 | grep -v "Too many"
  echo "== $m ($n vectors)"; vvp sim_q4_$m.vvp | grep -E "cnn_core_tb|PASS|FAIL|MISMATCH"
done
