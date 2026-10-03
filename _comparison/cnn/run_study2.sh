#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
E="--epochs 30 --size 40 --norm global"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
( run s2_c2 $E --convs 5-8-1,3-16-0,3-16-1 --fcs 32 ; run s2_c4 $E --convs 5-8-1,5-24-1 --fcs 48 ) &
( run s2_c3 $E --convs 5-16-1,3-32-0,3-32-1 --fcs 64 ; run s2_big $E --convs 5-32-1,3-64-0,3-64-1,3-128-0 --fcs 256 ) &
wait
