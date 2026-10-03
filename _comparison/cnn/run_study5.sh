#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
R="--frag 2 --neg-ratio 40 --epochs 40 --norm global"
V1="--convs 5-8-1,5-16-1 --fcs 32"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s5_big $R --size 40 --convs 5-32-1,3-64-0,3-64-1,3-128-0 --fcs 256 ) &
( run s5_x48 $R $V1 --size 48 ; run s5_x56d2 $R $V1 --size 56 --down 2 ) &
( run s5_x64d2 $R $V1 --size 64 --down 2 ; run s5_x40_c3 $R --size 40 --convs 5-16-1,3-32-0,3-32-1 --fcs 64 ) &
wait
