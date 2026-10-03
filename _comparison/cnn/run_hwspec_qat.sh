#!/bin/bash
cd /f/Projects/CFAR/_comparison/cnn
export HWDATA=hwspec HWGATE=0.75
( python quant_hw.py qat ../Results/hw/h_deep_s1.pt  --epochs 8 --jitter 2 --export ../Results/cnn_weights_hw/hw_deep  > logs/qat_h_deep.log 2>&1 ) &
( python quant_hw.py qat ../Results/hw/h_small_s1.pt --epochs 8 --jitter 2 --export ../Results/cnn_weights_hw/hw_small > logs/qat_h_small.log 2>&1 ) &
wait
python quant_hw.py qat ../Results/hw/h_xl_s1.pt --epochs 8 --jitter 2 --export ../Results/cnn_weights_hw/hw_xl > logs/qat_h_xl.log 2>&1
