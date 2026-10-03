#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
R="--frag 2 --neg-ratio 40 --epochs 60 --norm global --size 64 --down 2 --jitter 4"
D="--convs 5-16-1,3-32-0,3-32-1 --fcs 64"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s9_small_e60 $R --convs 5-8-1,5-16-1 --fcs 32 ; run s9_small_e60_s2 $R --convs 5-8-1,5-16-1 --fcs 32 --seed 2 ) &
( run s9_deep_e60_s2 $R $D --seed 2 ; run s9_deep_e60_s3 $R $D --seed 3 ) &
wait
