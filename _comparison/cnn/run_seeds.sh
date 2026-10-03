#!/bin/bash
# Seeds and splits for the confidence intervals: for every (split, seed) train, one after the other (RAM limit),
#   context float -> context INT8 (train_q3), plain float -> plain INT8 (train_q).     Logs: logs/seeds_<tag>.log
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=pooldetfxA CTXDIR=/f/Projects/CFAR/_comparison/Results/hw_cache_pooldetfxA SIDEFILE=side_codes.npy
run() {   # tag split seed
  tag=$1; export HWSPLIT=$2; seed=$3
  python train_ctx.py --name ctx_f_$tag --ctx 1 --side 1 --mil 1 --hard 1 --seed $seed > logs/seeds_ctx_f_$tag.log 2>&1
  python train_q3.py --src ctx_f_$tag --name ctx_q_$tag --epochs 8 --seed $seed > logs/seeds_ctx_q_$tag.log 2>&1
  python train_ctx.py --name pl_f_$tag --ctx 0 --side 0 --mil 1 --hard 1 --seed $seed > logs/seeds_pl_f_$tag.log 2>&1
  python train_q.py --src pl_f_$tag --name pl_q_$tag --epochs 8 --seed $seed > logs/seeds_pl_q_$tag.log 2>&1
  echo "done $tag" >> logs/seeds_progress.log
}
run sd2 20260925 2
run sd3 20260925 3
run sp2 20261004 1
run sp3 20261005 1
echo ALLDONE >> logs/seeds_progress.log
