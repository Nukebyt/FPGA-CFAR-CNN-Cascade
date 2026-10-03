# FPGA CFAR + CNN Cascade

Weibull-CFAR prescreen followed by a lightweight INT8 CNN, implemented on a Terasic DE10-Standard (Cyclone V 5CSXFC6D6F31C6) and evaluated on HRSID (5,604 images, 16,951 ships).

## Layout
| Path | Contents |
|---|---|
| `rtl/prescreen/` | integer pooled Weibull prescreen (bit-exact against the Python golden model), testbenches, regression |
| `rtl/cnn/` | INT8 CNN core (single tower and fine + context), RTL generators, golden-vector testbenches |
| `rtl/ctx/` | context-tower cascade (event RAM, context fetch, side-feature unit), JTAG wrapper |
| `rtl/cascade/` | single-tower two-clock cascade, pooled store, JTAG wrapper |
| `rtl/common/`, `lut/`, `_common/`, `CFAR_Weibull/` | shared Weibull CFAR building blocks and MATLAB reference |
| `_quartus/` | Quartus 21.1 projects and board-ready `.sof` files (`prescreen_probe`, `cascade_ps_jtag`, `cascade_ctx_jtag`) |
| `_comparison/fixedpoint/` | integer golden models, whole-dataset evaluation, confidence-interval scripts |
| `_comparison/cnn/` | training (float and INT8 QAT), export, board sweep host scripts (`sweep250/board/`) |
| `_comparison/Results/` | result tables, the INT8 models of every training run, and the raw on-board sweep (`sweep_ctx_all/frames.zip`) |
| top-level `*.md`, `*.docx` | design notes, studies, roadmaps and the comparison report with HRSID literature |

## Results in brief
Whole HRSID data set, measured on the board (context cascade, 1,015,171 events, bit-exact to the golden model):

| System | Ship recall | False events / image |
|---|---|---|
| Weibull prescreen only | 99.6 % | 166 |
| Weibull + INT8 CNN (97 % validation target) | 96.7 % (test split 96.2 %) | 2.3 (test split 2.6) |

The prescreen takes 9.01 ms per 800 x 800 frame at 100 MHz; the cascade fits in 11.9 k ALM, 394 M10K and 96 DSP.

## Not included
The HRSID dataset (obtain it from the dataset authors), cached patch tensors and per-image score arrays (regenerate with `_comparison/fixedpoint/extract_fx_events.py`), Quartus `db/` folders and simulation logs.

Start with the fixed-point and RTL design note (the top-level `*FIXEDPOINT_RTL_PRESCREEN*.md`) and `rtl/prescreen/README.md`.
