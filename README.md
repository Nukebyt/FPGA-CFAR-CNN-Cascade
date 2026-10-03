# FPGA CFAR + CNN Cascade (Paper 2)

Weibull-CFAR prescreen followed by a lightweight INT8 CNN, implemented on a Terasic DE10-Standard (Cyclone V 5CSXFC6D6F31C6) and evaluated on HRSID (5,604 images, 16,951 ships).

## Layout
| Path | Contents |
|---|---|
| `rtl/prescreen/` | integer pooled Weibull prescreen (bit-exact against the Python golden model), testbenches, regression |
| `rtl/cnn/` | INT8 CNN core (single tower and fine+context), RTL generators, golden-vector testbenches |
| `rtl/ctx/` | context-tower cascade (event RAM, context fetch, side-feature unit), JTAG wrapper |
| `rtl/cascade/` | single-tower two-clock cascade, pooled store, JTAG wrapper |
| `rtl/common/`, `lut/`, `_common/`, `CFAR_Weibull/` | shared Weibull CFAR building blocks and MATLAB reference |
| `_quartus/` | Quartus 21.1 projects and board-ready `.sof` files (`prescreen_probe`, `cascade_ps_jtag`, `cascade_ctx_jtag`) |
| `_comparison/fixedpoint/` | integer golden models, whole-dataset evaluation, confidence-interval scripts |
| `_comparison/cnn/` | training (float and INT8 QAT), export, board sweep host scripts (`sweep250/board/`) |
| `_comparison/Results/` | small result tables and the final INT8 model files |
| `PAPER2_*.md/.docx` | draft, roadmap, study notes, literature comparison |

## Not included
The HRSID dataset (obtain from the dataset authors), cached patch tensors and per-image score arrays (regenerate with `_comparison/fixedpoint/extract_fx_events.py`), Quartus `db/` folders and simulation logs.

Start with `PAPER2_FIXEDPOINT_RTL_PRESCREEN_2026-10-03.md` and `rtl/prescreen/README.md`.
