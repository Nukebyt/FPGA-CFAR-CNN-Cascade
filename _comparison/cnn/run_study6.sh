#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=64 HWGATE=0.45
R="--frag 2 --neg-ratio 40 --epochs 40 --norm global"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s6_2t $R --size 32 --convs 5-8-1,5-16-1 --size2 64 --down2 2 --convs2 5-8-1,5-16-1 --fcs 32 ; run s6_x48b $R --size 48 --convs 5-8-1,5-16-1 --fcs 32 ) &
( run s6_big64 $R --size 64 --convs 5-16-1,3-32-0,3-32-1,3-64-0 --fcs 128 ) &
wait
