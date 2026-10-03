# Cascade board bring-up (Paper 2 project; separate from the Paper 3 multi-model bring-up in ../../DE10_BRINGUP.md)


## Weibull prescreen + INT8 CNN cascade board demo -- BUILT 2026-10-02, RUN ON THE BOARD 2026-10-03 (default switches matched simulation: HEX5..3=018, HEX2..0=007, LEDR[9:6]=0111; programmed via `quartus_pgm -m jtag -c "DE-SoC [USB-1]" -o "p;output_files/cascade_de10_top.sof@2"`)

A different kind of bitstream from the five detector demos above: the full two-stage cascade
(`rtl/cascade/`, `rtl/cnn/`; README in `rtl/cascade/README.md`), self-contained like the others -- a 128x128
HRSID crop (`cascade_demo_image.hex`, 16 kB) sits in on-chip ROM and is streamed through Weibull -> trigger/gate
-> pooled frame store, then every candidate event is classified by the CNN, then the result is shown for ~2 s
and the loop repeats.

Project: `_quartus/cascade_de10/` (`quartus_sh --flow compile cascade_de10_top`; `.sof` in `output_files/`).

| Control | Function |
|---|---|
| SW[1:0] | Weibull Pfa plane (0: 1e-3 -- the CNN's training point; 1: 1e-4; 2: 1e-5; 3: 1e-6) |
| SW[3:2] | prescreen gate x - c1 >= tau (0: 0.60, 1: 0.70, 2: 0.75 -- trained, 3: 0.80) |
| SW[5:4] | CNN operating point = val-selected threshold for 80 / 85 / 90 / 95 % ship retention |
| KEY[0] | reset |

| Output | Meaning |
|---|---|
| LEDR[0] | at least one candidate accepted by the CNN in the last frame |
| LEDR[1] / [2] / [3] | frame finished pulse / streaming pixels / heartbeat |
| LEDR[4] / [5] | CNN stage busy / event FIFO overflowed (sticky) |
| LEDR[9:6] | accepted count (saturates at 15) |
| HEX5..3 / HEX2..0 | candidate events / CNN-accepted events this frame (hex) |

Quartus 21.1 (board wrapper, DEEP model, quad-pixel CNN core on a 100 MHz PLL clock): 25,305 ALM (60%), 239 M10K (43%), 78 DSP (70%); setup slack
+1.16 ns (100 MHz CNN domain) / +2.20 ns (50 MHz stream domain) at 85C, hold +0.21 / +0.24 ns, TNS 0. The same cascade at 800x800 (no wrapper,
`_quartus/cascade_deep/`): 21,866 ALM (52%), 428 M10K (77%), 47 DSP, setup +2.10 / +1.99 ns.

Physical bring-up checklist: (1) program the `.sof` over JTAG (`quartus_pgm`), (2) LEDR[3] blinks, (3) LEDR[2] flashes
briefly once per loop, (4) after the CNN finishes (about `events x 1.4 ms`, i.e. a few seconds for this crop) the frame
pulse LEDR[1] fires and HEX shows the counts, (5) at the default switches (SW all 0 except SW[3]=1, SW[4]=1, i.e. tau 0.75 and the 85% operating point) the simulation shows
**HEX5..3 = 018 (24 candidate events) and HEX2..0 = 007 (7 accepted)**, LEDR[9:6] = 0111; this is simulation-verified for both
the SMALL and DEEP models (`rtl/cascade/sim_de10_small.log`, `sim_de10_deep.log`, checked against `tb/check_cascade.py`).
A different switch setting changes the counts (tau 0.60/0.70/0.75/0.80 -> 37/29/24/17 candidate events).
Frame time on the board is dominated by the CNN: `cnn_core_q4` is ~19.7 k cycles/patch for DEEP at 100 MHz (~0.2 ms per event);
the 128x128 demo crop with 24 events finishes in a few milliseconds, so the result appears almost immediately after the frame streams
(the old serial-core estimate of `events x 1.4 ms` no longer applies). A PLL that fails to lock leaves the design in reset (all LEDs but the heartbeat dark).


**Note 2026-10-03:** the demo source now uses the layout-corrected tables (`genq_hw_deep_T`); the `.sof` built before the fix is kept as `output_files/cascade_de10_top_PRE_LAYOUT_FIX.sof`. The default-switch simulation result is unchanged (24 candidates / 7 accepted). Rebuilt and flashed 2026-10-03 (Quartus needs `SYNTH_TIMING_DRIVEN_SYNTHESIS OFF` with the T tables): 24,443 ALM (58%), 225 M10K (41%), 78 DSP (70%), setup +1.23 ns (100 MHz) / +3.39 ns (50 MHz), hold >= +0.22 ns. The 250-image sweep uses the separate JTAG-fed design (`jtag/README.md`).
