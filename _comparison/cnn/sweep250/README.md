# sweep250: cascade vs Weibull-only on 250 held-out HRSID test images (Paper 2)

Software, bit-exact model of the hardware (NOT run on the FPGA): MATLAB fixed-point Weibull (`../../sweep250_extract.m`)
+ Python integer-exact INT8 CNN (`quant_hw.int_forward`). Definitions are in the docstring of `sweep250.py`.

```
# 1. image list (250 evenly spaced images of the 842-image test split)
python sweep250.py --make-list
# 2. fixed-point Weibull detection maps, 4 Pfa planes (~1 min)
matlab -batch "addpath('F:/Projects/CFAR'); cfar_setup(); cd('F:/Projects/CFAR/_comparison'); sweep250_extract"
# 3. triggers -> gate -> CNN -> pixel-level Pd/Pfa, bootstrap CIs (~3 min)
python sweep250.py --score
# 4. figures only (edit the PLOT section of sweep250.py, re-run; seconds)
python sweep250.py --plot
python validate_events.py     # check: rebuilt plane-0 events == earlier MATLAB hwspec extraction (5378/5378)
```
Edit the CONFIG block at the top of `sweep250.py` for the model (DEEP/SMALL/XL), gate tau, operating points, image count.
Outputs in `_comparison/Results/sweep250/`: `summary.json`, `counts.npz`, `events.npz`, `fig_pd_bars.png`,
`fig_pfa_log.png`, `fig_fa_events_log.png`, `fig_pd_vs_pfa.png`.

## On-board run (JTAG-fed cascade)
```
# program _quartus/cascade_jtag/output_files/cascade_jtag_top.sof (see rtl/cascade/jtag/README.md), then:
cd board && python board_sweep.py --n 250        # 1000 frames, ~5 min; results in Results/sweep250/hw/
cd .. && python sweep250.py --score --hw --plot  # -> counts_hw.npz, summary_hw.json, fig_*_hw.png, hw_vs_model.json (0 mismatches expected)
```
`board/board.py` is the host driver (System Console TCP server `jtag_server.tcl` + register access); `sweep250.py --hw` takes the CNN logits from the board.
