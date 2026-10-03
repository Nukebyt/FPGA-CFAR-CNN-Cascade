#!/bin/bash
# Capacity ladder at the best context (64x64 patch, 2x2 down-sampled -> 32x32 net), trained WITH offset jitter.
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
R="--frag 2 --neg-ratio 40 --epochs 40 --norm global --size 64 --down 2 --jitter 4"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s7_v1   $R --convs 5-8-1,5-16-1 --fcs 32 ; run s7_deep $R --convs 5-16-1,3-32-0,3-32-1 --fcs 64 ) &
( run s7_wide $R --convs 5-16-1,5-32-1 --fcs 64 ; run s7_v1_s2 $R --convs 5-8-1,5-16-1 --fcs 32 --seed 2 ) &
wait
