#!/bin/bash
# Final models on full HRSID (5,604 images; stored patches are already the deployed 2x2-pooled 32x32 input).
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=full32 HWGATE=0.45
R="--frag 2 --neg-ratio 16 --epochs 40 --norm global --size 32 --down 1 --jitter 2"
D="--convs 5-16-1,3-32-0,3-32-1 --fcs 64"
S="--convs 5-8-1,5-16-1 --fcs 32"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run f_deep_s1 $R $D --seed 1 ; run f_deep_s2 $R $D --seed 2 ) &
( run f_small_s1 $R $S --seed 1 ; run f_deep_s3 $R $D --seed 3 ) &
wait
