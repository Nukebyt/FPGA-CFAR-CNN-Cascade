#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
R="--frag 2 --neg-ratio 40 --epochs 40 --norm global --size 64 --down 2 --jitter 4"
D="--convs 5-16-1,3-32-0,3-32-1 --fcs 64"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s8_deep_s2 $R $D --seed 2 ; run s8_d3 $R --convs 5-12-1,3-24-0,3-24-1 --fcs 48 ; run s8_deep_e60 $R $D --epochs 60 ) &
( run s8_deep_s3 $R $D --seed 3 ; run s8_d2 $R --convs 5-16-1,3-32-0,3-32-1,3-48-0 --fcs 64 ; run s8_deep_h7 $R $D --hard-frac 0.75 --hard-top 0.03 ) &
wait
