# Pooled-domain Weibull prescreen — fixed-point model and RTL (Paper 2)

Status 2026-10-03. Golden model, RTL (`prescreen_core.v`, `peak5.v`, `prescreen_top.v`, `cascade_ps_2clk.v`) and bit-exact simulation are in place.
The numbers below are measured; anything not measured is listed under "Open items".

## What it computes

Config A of the Pd study (`PAPER2_PD_ROADMAP…md` §11): **2×2 pooled** log image, ring window **25/17** (N = 336 cells, 4 pixels thick on the pooled grid, 8 px at full resolution),
Pfa selectable from {3e-2, 1e-2, 1e-3, 1e-4}, gate on the contrast, **5×5 contrast-peak events**, mirrored (symmetric) image border.

Everything is integer (`_comparison/fixedpoint/prescreen_fx.py` is the golden model, the RTL is bit-exact to it):

| quantity | definition | width |
|---|---|---|
| `q8` | `QROM[pixel]` = round(255·(½ln(I+½) − XLo)/(XHi − XLo)), XLo = −0.40, XHi = 2.80 (step 0.01255 nat) | 8 u |
| `P` | `(q00+q01+q10+q11+2)>>2` – the pooled frame the CNN patch store already holds | 8 u |
| `S1`, `S2` | ring sums of `P` and `P²` (25×25 window minus 17×17 guard) | 17 u, 25 u |
| `A` | `N·P − S1` (= N × contrast) | 18 s |
| `num` | `N·S2 − S1²` (= N(N−1)·c₂, exact) | 34 u |
| decision | `A > 0` and `A² > ((clamp(num, NUM_LO, NUM_HI) >> 12) · KC[pfa])` | 34 u |
| gate | `A ≥ G`, `G = round(N·τ/step)` (τ = 0.6 → 16065) | 17 u |
| event | gated pixel with maximum `A` in its 5×5 neighbourhood (ties kept) | – |

`KC = round(κ²·N/(N−1)·2¹²)` with κ = K(Pfa)·√6/π and K = γ + ln(−ln Pfa). The Weibull shape clamp C ∈ [0.8, 8] becomes a clamp on `num` (`NUM_LO`, `NUM_HI`).
**No ROM, no square root and no divider**: the float rule `x > c₁ + K/C` with `C = π/√(6c₂)` is the same inequality as `A² > κ²·(N/(N−1))·num` for `A > 0`.

## Architecture (two-pass cascade)

```
pass 1  clk 50 MHz     pixel stream ─QROM─ 2x2 pool ──► pooled store (400x400x8)      pool_store_writer.v (gap tolerant, unchanged)
pass 2  clk_cnn 100 MHz prescreen_top reads the stored frame, mirrored border:           prescreen_core.v + peak5.v
                         events (j,i) ──► event RAM
pass 3  clk_cnn         per event: patch_fetch → cnn_core → logit ≥ θ                  unchanged
```
Because the frame is stored before the prescreen runs, the streaming Weibull core (gap-intolerant line buffers, 8-px unevaluated border) and `trigger_gate` are gone,
and a slow host can feed pixels directly (no clock gating).

`prescreen_core`: per padded pixel (r, c) five reads of the store (one per clock): `Pp(r,c)`, `Pp(r−25,c)`, `Pp(r−4,c)`, `Pp(r−21,c)` and the centre `Pp(r−12,c−12)`.
Vertical running sums per column (RAMs, x and x²) for the 25-row window and the 17-row guard; horizontal running sums by shift registers of the new column sums.
`peak5`: horizontal 5-maximum (shift register) + four column-indexed RAMs holding the previous four rows' maxima; the centre pixel's contrast and payload are delayed two rows through two more RAM chains.
The module injects its own zero steps at row ends and after the last row, so the stream may stop at the last real pixel.

Throughput: (400+24)·(400+24)·5 = 0.90 M clk_cnn cycles per frame (9.0 ms at 100 MHz), independent of the image content.

## Files

| file | role |
|---|---|
| `prescreen_core.v` | sequencer, mirrored addressing, running sums, A / num / decision / gate |
| `peak5.v` | 5×5 contrast-peak events, raster order, payload `{S1, num>>12, P}` carried to the event |
| `prescreen_top.v` | core + peak5 |
| `cascade_ps_2clk.v` | two-clock cascade with this prescreen (store, event RAM, patch fetch, CNN, handshake) |
| `tb/prescreen_tb.v`, `tb/check_prescreen.py` | prescreen alone vs the Python golden model, two frames back to back |
| `tb/cascade_ps_tb.v`, `tb/check_cascade_ps.py` | whole cascade: store, events and CNN logits vs Python |
| `build_cascade_ps.sh` | iverilog build line for the cascade test |
| `../../_comparison/fixedpoint/` | golden model, whole-data-set evaluation, event extraction for CNN training |

## Verification and results (2026-10-03, later)

* Prescreen alone: `tb/run_regression.sh`, 6/6 bit-exact (events j, i, A, S1, num>>12, P) vs `_comparison/fixedpoint/prescreen_fx.py`.
* Cascade with the INT8 CNN tables `../cnn/genq_fx_plain` (retrained on hardware-exact candidates, `_comparison/cnn/train_q.py`): `build_cascade_ps.sh` + `tb/check_cascade_ps.py` — store, events, logits, accept flags all exact; second frame identical (192 x 192 crop with gaps in the pixel feed: 54 events).
* JTAG wrapper (`jtag/cascade_ps_jtag_core.v`, register map in its header; no clock gate): `build_jtag_ps_tb.sh`, register-level flow with random host gaps, second frame identical.
* Quartus (`_quartus/cascade_ps_jtag`): 11,172 ALM (27 %), 397 M10K (72 %), 74 DSP, timing met at 100 MHz (+1.49 ns) and 50 MHz (+3.98 ns).
* Host: `_comparison/cnn/sweep250/board/board_ps.py`, `board_sweep_ps.py` (per-frame bit-exact check against the Python models). Gate G = 16065, KC table, theta = -17457 (97 % retention).
