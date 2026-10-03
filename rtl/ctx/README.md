# Context-tower cascade (Paper 2)

Pooled Weibull prescreen (`../prescreen/`) + INT8 context CNN: fine tower + context tower (4x4-pooled 128x128-px field of view) + 9 hardware-exact side codes.
See `PAPER2_FIXEDPOINT_RTL_PRESCREEN_2026-10-03.md` section 5 for results.

| file | role |
|---|---|
| `cascade_ctx_2clk.v` | top: pass 1 pool + store + image statistics, pass 2 prescreen + side unit + event RAM, pass 3 per event: 9 side bytes, fine patch, context patch -> `cnn_core_ctx` |
| `ctx_fetch.v` | context patch (4,096 store reads: 2x2 pool of the pooled frame, edge replicate) |
| `side_unit.v`, `isqrt_pipe.v` | per-event side codes f0..f3 |
| `img_stats.v`, `img_codes.v`, `ln_rom.v` | image-level codes f4..f8 |
| `jtag/cascade_ctx_jtag_{core,top}.v` | JTAG-fed wrapper (same register map as the prescreen design; ID 0xCA5CADE4) |
| `../cnn/cnn_core_ctx.v`, `../cnn/gen_cnn_rtl_ctx.py` | CNN core with three input streams / segments, table generator |
| `build_cascade_ctx.sh`, `tb/cascade_ctx_tb.v`, `check_cascade_ctx.py` | system simulation and independent check |

Golden models: `_comparison/fixedpoint/side_fx.py`, `_comparison/cnn/quant_ctx.py` (integer network), training `_comparison/cnn/train_q3.py`.
Board: `_comparison/cnn/sweep250/board/board_ctx.py`, `board_sweep_ctx.py` (bitstream `_quartus/cascade_ctx_jtag/output_files/cascade_ctx_jtag_top.sof`; theta -14462 = 97 % retention).
