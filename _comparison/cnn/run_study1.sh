#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
A="--convs 5-8-1,5-16-1 --fcs 32 --epochs 30"
run() { python train_hw.py --name "$1" ${@:2} > logs/$1.log 2>&1; }
mkdir -p logs
( run s1_g32  $A --size 32 --norm global --jitter 4 ; run s1_c32 $A --size 32 --norm c1 --jitter 4 ; run s1_g40 $A --size 40 --norm global ) &
( run s1_p32  $A --size 32 --norm pmean  --jitter 4 ; run s1_p40 $A --size 40 --norm pmean ; run s1_c40 $A --size 40 --norm c1 ) &
wait
