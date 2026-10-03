#!/bin/bash
# Final models on the hardware-faithful dataset (NMS trigger, pooled frame store, gate = x - c1 >= 0.75).
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=hwspec HWGATE=0.75
R="--frag 2 --neg-ratio 16 --epochs 40 --norm global --size 32 --down 1 --jitter 2"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
run h_deep_s1  $R --convs 5-16-1,3-32-0,3-32-1 --fcs 64 --seed 1
run h_small_s1 $R --convs 5-8-1,5-16-1 --fcs 32 --seed 1
run h_xl_s1    $R --convs 5-24-1,3-48-0,3-48-1 --fcs 96 --seed 1
