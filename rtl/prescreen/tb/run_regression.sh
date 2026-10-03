#!/bin/bash
# Prescreen regression: prescreen_top vs the Python golden model on real HRSID crops (several window geometries / Pfa planes / sizes).
# usage: bash tb/run_regression.sh      (about 5 minutes; three simulations run at a time)
cd "$(dirname "$0")"
run() { python check_prescreen.py "$@" 2>&1 | grep -E "pooled|BIT|MISMATCH|only|second" | tr '\n' ' '; echo; }
( run --img P0001_0_800_9000_9800.png --y0 600 --x0 40 --h 200 --w 200 --sli 25 --guard 17 --pfa 0 --tau 0.6 --out out_a ) &
( run --img P0001_1200_2000_10190_10990.png --y0 500 --x0 100 --h 300 --w 160 --sli 17 --guard 13 --pfa 1 --tau 0.75 --out out_b ) &
( run --img P0001_0_800_8400_9200.png --y0 400 --x0 200 --h 250 --w 250 --sli 25 --guard 17 --pfa 3 --tau 0.6 --out out_c ) &
wait
( run --noise 5 --h 60 --w 80 --sli 25 --guard 17 --pfa 2 --tau 0.6 --out out_d ) &
( run --img P0001_1200_2000_3600_4400.png --y0 560 --x0 200 --h 240 --w 200 --sli 21 --guard 15 --pfa 1 --tau 0.6 --out out_e ) &
( run --img P0001_0_800_10190_10990.png --y0 0 --x0 150 --h 160 --w 300 --sli 25 --guard 17 --pfa 2 --tau 0.6 --out out_f ) &
wait
