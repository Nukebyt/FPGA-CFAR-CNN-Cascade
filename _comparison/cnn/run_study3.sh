#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
B="--convs 5-8-1,5-16-1 --fcs 32 --size 40 --norm global"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s3_all $B --frag all --epochs 30 ; run s3_k2 $B --frag 2 --neg-ratio 40 --epochs 40 ) &
( run s3_k1 $B --frag 1 --neg-ratio 64 --epochs 40 ; run s3_k3 $B --frag 3 --neg-ratio 32 --epochs 40 ) &
wait
