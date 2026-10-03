# Weibull-CFAR + CNN cascade — RTL (DE10-Standard, Cyclone V 5CSXFC6D6F31C6)

Status 2026-10-03: simulation-verified (DEEP tables `genq_hw_deep_T`, see the layout fix in the study §10; JTAG-fed 800x800 flow in `jtag/README.md`), Quartus-fit at 800×800, and **run on the DE10-Standard** (default switches matched simulation: 24 candidates / 7 accepted). No power number yet.

## What it does

```
pixel stream ──► weibull_front_cascade (verified Weibull core + aligned x, c1)
      │                    │
      │                    └─► trigger_gate ──► event_fifo   (pass 1: streams once, 1 px/clk)
      │                          NMS trigger + gate (x−c1 ≥ τ)
      └─► pool_store_writer ──► pooled_store (¼-size frame: QROM → 2×2 average)

after the frame has streamed (pass 2, per event):
   event_fifo ─► patch_fetch (32×32 window, edge-replicated, from pooled_store) ─► cnn_core ─► logit ≥ θ ?
```
The CNN is ~20–100× slower than the pixel stream, so the frame is stored (pooled, 160 kB at 800×800) and
the events are queued; the CNN runs after the frame.

## Spec (the training data was built to this exact definition — `_comparison/extract_cnn_patches_hwspec.m`)

1. `q8 = QROM[pixel] = floor(clip((½·ln(I+0.5) − XLo)/(XHi−XLo),0,1)·255 + ½)`, XLo=−0.40, XHi=2.80 (`qrom.hex`, `gen_qrom.py`)
2. `P[j][i] = (q[2j][2i]+q[2j][2i+1]+q[2j+1][2i]+q[2j+1][2i+1]+2) >> 2` on the fixed frame grid
3. Weibull detection `D` (Pfa plane via `pfa_sel`; 0 → 1e-3)
4. trigger `T = D & ~D(y,x−1) & ~D(y−1,x−1) & ~D(y−1,x) & ~D(y−1,x+1)` — needs only the previous row of detect bits
5. gate `g = x − c1 ≥ τ` (Q.14: `x_code + 19866 − 2·c1_code ≥ tau_q14`)
6. event `(j,i) = (y>>1, x>>1)`; window `P[j−16..j+15][i−16..i+15]`, indices clamped (edge replicate)
7. `cnn_core` → integer logit; accept if `logit ≥ theta`

Quirk inherited from the Weibull core: it never emits the first interior pixel (`CMP_OFFSET=1` in
`weibull_top_tb.v`), so `trigger_gate` starts its column counter at 1.

## Modules

| file | role |
|---|---|
| `../cnn/cnn_core.v` | INT8 layer-sequential MAC array (L lanes), bit-exact to `cnn/quant_hw.py int_forward` |
| `../cnn/gen_cnn_rtl.py` | exported model → `cnn_w.hex`, `cnn_pq.hex`, `cnn_cfg.vh` (`gen_{small,deep,xl}/`) |
| `weibull_front_cascade.v` | `weibull_top_new` internals + x/c1 aligned with `detect` (original file untouched) |
| `trigger_gate.v` | NMS trigger + gate + pooled-grid coordinates |
| `pool_store_writer.v`, `pooled_store.v` | QROM + 2×2 pooling + frame store |
| `event_fifo.v` | candidate queue (overflow flag is sticky) |
| `patch_fetch.v` | 32×32 window fetch with edge clamp, streams into the CNN |
| `cascade_top.v` | controller (two-pass), frame-level reset of the Weibull core between frames |

## Verification (all with iverilog; `build_casc.sh`, `tb/`)

| test | result |
|---|---|
| `cnn/tb/cnn_core_tb.v` vs exported golden logits | SMALL 32/32, DEEP 6/6 vectors **bit-exact** (more run in the background log) |
| `tb/cascade_tb.v` + `tb/check_cascade.py` (128×128 HRSID crop, SMALL) | pooled store exact; 37/37 trigger+gate events match a Python recomputation from the RTL's own detect/x/c1 stream; 37/37 CNN logits bit-exact; 1.54 M cycles |
| `tb/cascade_de10_top_tb.v` (board wrapper, default switches, SMALL hw-faithful model) | store exact; 24/24 events and 24/24 logits verified; display shows 0x018 candidates, 0x007 accepted (`sim_de10_small.log`). DEEP regression: `sim_de10_deep.log` |

## Quartus 21.1, 800×800, DEEP model (`_quartus/cascade_deep/`)

| | value |
|---|---|
| ALMs | 21,866 / 41,910 (52%) |
| M10K | 428 / 553 (77%) |
| DSP | 47 / 112 (42%) |
| Fmax / slack @50 MHz | setup +2.10 ns (85C) / +1.99 ns (0C); hold +0.18 / +0.08 ns; TNS 0 |

CNN core alone (`_quartus/cnn_core_deep/`): 1,562 ALMs, 147 M10K, 33 DSP, Fmax 93.7 MHz.

## Performance of the ORIGINAL serial core at 50 MHz (superseded by the throughput upgrade below)

`cnn_core` at L=32: SMALL 40.1 k cycles/patch, DEEP 71.0 k, XL ≈ 172 k (issue-bound; ~1 MAC lane per
DSP). At ~275 gated events/image this is ~0.1–0.8 s/frame — the CNN, not the Weibull stream, is the bottleneck.

## Known limits / open items

- Weibull stream must be gap-free within a frame (KNOWN_ISSUE_GAP_INTOLERANCE); use `frame_buffer_bridge` for live HPS data.
- Edge handling is replicate (clamp); training used the same.
- Board run done 2026-10-03 (default switches only; tau sweep not yet checked on hardware); no power number.
- CNN throughput: done, see "Throughput upgrade" below.
- `cascade_tb` simulation of one 128×128 crop took ~7 minutes in iverilog; full 800×800 frames are not practical
  to simulate — use crops, and the Python/MATLAB models for whole-scene numbers.


## Hardware-exact end-to-end numbers (INT8 models, 842 test images, thresholds from validation)

Bit-exact fixed-point Weibull detection + this trigger/gate + pooled-store patches (`_comparison/extract_cnn_patches_hwspec.m`
with `Fixed=true`, `_comparison/cnn/eval_hwexact.py`): 1,630 triggers/img, 274 gated events/img, 89.3% of GT ships reachable.
At the 90% validation operating point: XL 88.0% ship retention at 1.74 FA/img, DEEP 87.9% at 1.98, SMALL 86.8% at 2.71
(CFAR alone: ~1,600 FA/img) — ~15 candidates/img out. These agree with the float-pipeline test split, so the float-derived
training data is representative of what this RTL produces. Details and CIs: `PAPER2_CNN_HW_STUDY_2026-10-02.md` §9.

## Bug found while verifying the board wrapper

`cascade_de10_top` first overwrote `pixel_in_valid` in the same cycle it drove the frame's last pixel, so the last pixel
was never streamed, `frame_in_done` never fired, and the controller waited forever (found because the wrapper simulation
hung). Fixed. The older per-detector `*_de10_top.v` wrappers use the same sequencer pattern and were not re-checked.


## Throughput upgrade (2026-10-02): quad-pixel core + 100 MHz CNN clock — 5.4–6.5× faster frames

Two independent changes, each verified bit-exact:

1. **`../cnn/cnn_core_q4.v`** (tables from `../cnn/gen_cnn_rtl_q4.py`, dirs `../cnn/genq_hw_*/`). Same interface, same arithmetic,
   but every pass computes a 2×2 *quad* of output pixels from one weight word (4×L MACs/cycle). Feature maps live in 4 parity
   banks (`bank = 2(y&1)+(x&1)`) so the four input pixels of any tap are read in the same cycle through a 4×4 byte crossbar.
   Pool layers take the max on the raw accumulators before requantisation (requant is monotone, so this is exact) — one requant
   per lane per quad. FC-after-conv is a K=H conv with a 1×1 output. Pixel lanes 0–1 use DSP multipliers, lanes 2–3 ALM multipliers
   (one DSP per 9×9 multiplier would exhaust the device). Pipelined requant walk + registered crossbar → **Fmax 113 MHz**.
2. **`cascade_top_2clk.v`**: the CNN stage (patch fetch, core, event reads, per-frame control) runs on a 100 MHz PLL clock
   (`cnn_pll.v`, `altera_pll` instantiated directly; `-DSIM_PLL` for simulation); the Weibull stream stays on 50 MHz. Crossings:
   dual-clock `pooled_store_dc`/`event_ram_dc` (never accessed by both sides at once) and a 4-phase level handshake
   (`go` →, ← `fin`, 2-FF synchronised); counts are sampled only while the other side is quiescent. The single-clock `cascade_top.v`
   is kept as the reference.

| | serial core @50 MHz | quad core @50 MHz | **quad core @100 MHz** |
|---|---|---|---|
| cycles/patch (measured, RTL sim) SMALL / DEEP / XL | 40.1k / 71.0k / (172k est.) | 10.4k / 19.7k / 48.2k | same, at 2× the clock |
| frames/s at 800×800, 274 events/frame: SMALL | 4.2 | 13.2 | **22.5** |
| DEEP | 2.5 | 7.9 | **14.4** |
| XL | 1.0 | 3.5 | **6.8** |

(`throughput.py`; the model reproduces the simulated two-clock frame time to 0.02%: 154,516 vs 154,491 cycles.)

**Verification of the upgrade:** `cnn_core_q4` bit-exact on the golden vectors — SMALL 64/64, DEEP 48/48, XL 12/12 (`../cnn/q4_golden.log`);
two-clock cascade (128×128 crop, SMALL): pooled store, 24/24 events and 24/24 logits match the Python reference; a **second frame** through
the same hardware is identical to the first (handshake/reset path reusable); board wrapper on the two-clock quad DEEP design passes the
same independent check (24 candidates, 7 accepted).

**Quartus 21.1, 800×800, DEEP, two clocks (`_quartus/cascade_2clk_deep/`):** 27,426 ALM (65%), 428 M10K (77%), 79 DSP (71%);
setup slack +2.94 ns (50 MHz domain) / +1.40 ns (100 MHz domain), hold ≥ +0.15 ns, TNS 0; PLL-derived clocks, domains declared asynchronous.
Quad core alone: 8.0k ALM, 148 M10K, 65 DSP, Fmax 113 MHz (`_quartus/cnn_core_q4_deep/`).
Board bitstream (`_quartus/cascade_de10/`, 128×128 demo): 25,305 ALM (60%), 239 M10K (43%), 78 DSP (70%), setup +1.16 / +2.20 ns.
**XL at 800×800 (two clocks, quad core, `_quartus/cascade_2clk_xl/`):** fits — 27,465 ALM (66%), **553/553 M10K (100%, no margin)**, 79 DSP; setup +1.07 ns (100 MHz) / +3.80 ns (50 MHz). Its 5,017-word weight ROM alone is ~300 M10K.
