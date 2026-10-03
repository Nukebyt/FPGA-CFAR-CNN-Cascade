#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
B="--convs 5-8-1,5-16-1 --fcs 32 --size 40 --norm global"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s4_k1_s2 $B --frag 1 --neg-ratio 64 --epochs 40 --seed 2 ; run s4_k1_s3 $B --frag 1 --neg-ratio 64 --epochs 40 --seed 3 ) &
( run s4_k1_ta $B --frag 1 --neg-ratio 64 --epochs 40 --train-all ; run s4_all_ta $B --frag all --epochs 30 --train-all ) &
wait
