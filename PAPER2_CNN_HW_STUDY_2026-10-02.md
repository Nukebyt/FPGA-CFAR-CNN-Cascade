# Paper 2 — CNN discriminator: hardware-constrained re-study (2026-10-02)

Companion to `PAPER2_ROADMAP_weibull-cnn-cascade.md`. **Read §1 first: three of the
roadmap's earlier numbers/assumptions were wrong, and this study replaces them.**

## 1. What was wrong, and what replaces it

| # | Earlier claim (roadmap / draft §3.3) | Finding | Evidence |
|---|---|---|---|
| 1 | "225,376 Weibull-flagged clusters over 1,200 images = 187.8 candidates/image; only 12.6% are real ships"; CNN gives 68.3% precision, **6.5× reduction** | **Not the real candidate stream.** `extract_cnn_patches.m` caps negatives at 8× positives per image. Weibull at Pfa=1e-3 really emits **~1,670 clusters/image, of which only 1–2% touch a ship**. Every precision / reduction figure computed on the capped set overstated the cascade. | 12-image MATLAB probe (sli=61: 14,386 neg + 288 pos; sli=17: 20,812 neg + 277 pos); 1,200-image extraction: 2,002,303 clusters, 23,533 ship-overlapping (1.18%); all-HRSID cluster count: mean 1,668/img |
| 2 | CNN trained on candidates from `sli=61/guard=49` | The DE10 RTL is **SLI=17/GUARD=13** (`front_end3.v`; FINDINGS §8: the window must stay small to fit Cyclone V next to anything else). The CNN must be trained on the candidate stream the *deployed* prescreen produces. | `rtl/common/front_end3.v` parameters; `_quartus/weibull*/` fits |
| 3 | Patch "fetched from the existing line-buffer window" | The Weibull window is 17×17; a CNN patch is ≥ 32×32. The patch needs **its own 8-bit row buffer** (64 rows × W). It costs M10K, so it is now in the budget. | `rtl/common/line_buffer.v`; `cnn/hwcost.py` |

Replacement: `extract_cnn_patches_hw.m` (hardware window, **every** cluster kept, uint8
patches = the CNN's real input precision, ship IDs recorded) and a **ship-level**
metric (§3). All v1–v3 numbers in the draft are superseded; they remain valid only as
"CNN-vs-CNN on the capped subsample" history.

## 2. Constraint recheck (DE10-Standard, 5CSXFC6D6F31C6)

Measured = Quartus 21.1 fit report. Derived = `cnn/hwcost.py` (calibrated: reproduces the
measured 70 M10K at W=64 and 118 at W=800). Estimate = engineering assumption, **not** a fit.

| Resource | Device | Weibull prescreen (**measured**, SLI=17, **W=800**) | Free | CNN stage + trigger, 128 MAC lanes (**derived/estimate**) |
|---|---|---|---|---|
| ALM | 41,910 | 17,411 (42%) | 24.5k | ≈ 29.6k total (71%) — *estimate* |
| M10K | 553 (5.66 Mb) | **118 (21%)** (70 at W=64) | 435 | SMALL 222 (40%) · DEEP 279 (50%) · XL 366 (66%) total. DEEP breakdown: Weibull rows 118, patch row buffer 64, weights ≈ 63 (incl. 20% packing margin), activations ≈ 22, ping-pong 8, trigger delay plane 4 |
| DSP | 112 (×3 9×9 each) | 13 (12%) | 99 → 297 int8 MAC lanes | 58 total (52%) at 128 lanes |
| Timing @ 50 MHz | — | setup +2.25 ns (85C) / +2.03 ns (0C), hold +0.15 / +0.01 ns, TNS 0 (W=800) | — | not yet built |

What this changes about "improve the model":

1. **The 16k-parameter ceiling was never the real limit.** Memory alone fits ≥ 300k INT8
   weights next to Weibull. The binding resource is *compute throughput at the true
   candidate rate*, not parameters, M10K or DSP.
2. **A free gate buys ~7× of that throughput.** (mean of the centre 3×3 of `fe.x`) − `c1` ≥ 0.45
   passes only **13.8% of candidates (231/img, test)**. `c1` is already computed by Weibull's
   moment engine. This is part of the cascade, not an extra model.
3. **Patches are tiny clusters, so full CCL is unnecessary.** Ship-overlapping clusters have
   median extent 2 px (p99 12 px); a first-pixel NMS trigger lands within ~1–3 px of the
   centroid. But offsets matter: a model trained on centroid-centred patches degrades
   **2.5× at a 3 px shift and 5× at 5 px** (measured). Jitter training is mandatory and
   fixes it (§4).
4. **Throughput is the real trade-off** (§5, §6): accuracy keeps improving with MACs, and MACs
   cost frames/second, not memory.

## 3. Metric (replaces per-cluster accuracy/precision)

A cascade is judged per ship and per image:
- **Ship retention** = fraction of CFAR-*reachable* ships (≥ 1 ship-overlapping cluster;
  90.6% of GT ships on the full-HRSID test split at this Pfa/SLI) with ≥ 1 accepted cluster. A
  ship breaks into ~4–7 fragments; only the best has to pass.
- **FA/image** = accepted false-alarm clusters per image (CFAR alone: ~1,650).
- Model selection on **validation mean FA/img over 70–90% retention**; the operating
  threshold is chosen on validation and applied unchanged to test (oracle-threshold test
  curves are optimistic and are shown only for comparison). Splits are by image.
- **End-to-end ship recall** = retention × reachable fraction (ships found / all GT ships).
- Reference: the free gate alone (centre-3×3 mean − `c1`) leaves **23.3 / 29.1 / 39.8 / 62.8 FA/img at 80 / 85 / 90 / 95% retention** (full-HRSID test). The CNN is judged against that, not against raw CFAR.

## 4. What moved the metric (ablations on the 1,200-image subset; 840 train / 180 val / 180 test images)

Gated universe (gate ≥ 0.45), validation mean FA/img over 70–90% retention; lower is better.
(Subset numbers; the full-HRSID run in §5 is the headline.)

| Change | Before → after (val mean FA/img) | Verdict |
|---|---|---|
| Train on the 2 best fragments per ship instead of all fragments (others ignored) | 1.73 → 0.85 (test 3.41 → 1.71) | **Largest single gain** (~2×); replicated over k=1,2,3 and 3 seeds |
| Context 40 px → 64 px (2×2-pooled to a 32×32 net input), *with fewer MACs* | 0.85 → 0.68 | **Gain** (−20%); 48 px full-res and a two-tower net are no better |
| Offset jitter ±4 px in training | shift-3 degradation 2.5× → 1.05× | **Required** for a streaming trigger |
| 3×3-stack depth (5×5-16 → 3×3-32 → 3×3-32) vs 5×5-8/16 | 0.63–0.71 → 0.55–0.58 (40 ep) | Gain, stable over 3 seeds |
| 40 → 60 epochs (deep) | 0.55–0.58 → 0.40–0.46 | Gain on val; test flat |
| Input normalisation (global / patch-mean / `c1`) | within noise | Neutral → use `global` (folds into layer 1) |
| 4× width (5×5-16/32, 1.64M MAC) | 0.63 → 0.66 | Neutral *on the subset* |
| Hard-negative over-weighting (75% hard, top 3%) | 0.55 → 1.98 | **Harmful** |
| Train on gate-passing only vs all clusters | within noise | Neutral |
| PTQ → INT8 (per-channel weights, uint8 acts) | +5–20% mean FA vs float | Acceptable |
| QAT fine-tune | ≈ PTQ | No extra gain (lr 2e-4 *hurt*; 2e-5 anchored at PTQ is neutral) |

Every run is in `Results/hw/results.jsonl`; `cnn/compare_runs.py` regenerates the comparison
with image-bootstrap CIs.

## 5. Final models — full HRSID, INT8, integer-exact

**Data.** All 5,604 HRSID images; 1,781,144 gated clusters (negatives below the 0.40 gate dropped,
every positive kept; 105,925 ship-overlapping); split by image 3,922 train / 840 val / 842 test.
Test: 2,877 GT ships, **2,606 CFAR-reachable (90.6%)**, 1,674 CFAR candidates/img, 231/img after the
gate. Stored patches are the deployed 2×2-pooled 32×32 uint8 input (the hardware buffers 64×64).

**Models** (all: BN folded, per-channel int8 weights, uint8 activations, 15-bit per-channel requant,
accumulators ≤ 22 bits, no hand features; training: 2 best fragments/ship, jitter ±4 px, 40 epochs):

| | SMALL | DEEP | XL |
|---|---|---|---|
| Layers | 5×5-8 → pool → 5×5-16 → pool → FC32 → FC1 | 5×5-16 → pool → 3×3-32 → 3×3-32 → pool → FC64 → FC1 | 5×5-24 → pool → 3×3-48 → 3×3-48 → pool → FC96 → FC1 |
| Params / MACs per patch | 16,289 / 0.49 M | 65,633 / 1.95 M | 147,217 / 4.15 M |
| Float test FA/img @ 80/85/90/95% (oracle thr.) | 1.1 / 1.9 / 3.4 / 7.8 | 0.5–0.6 / 1.0–1.1 / 1.9–2.2 / 4.7–5.0 (3 seeds) | 0.5 / 0.8 / 1.5 / 3.9 |

Deployed operating points — threshold chosen on **validation**, applied unchanged to the held-out
**test** split (842 images). End-to-end ship recall is of all 2,877 GT ships.

| Model | Val target | Test ship retention (95% CI) | FA/img (95% CI) | Candidates/img out | Reduction vs CFAR alone | End-to-end ship recall |
|---|---|---|---|---|---|---|
| XL | 80% | 77.4% (73.9–80.4) | 0.38 (0.29–0.46) | 9.5 | 176× | 70.1% |
| XL | 85% | 83.1% (80.0–85.8) | 0.78 (0.64–0.93) | 10.7 | 156× | 75.3% |
| XL | 90% | 89.6% (87.4–91.5) | **1.51 (1.25–1.75)** | 12.4 | 135× | **81.2%** |
| DEEP | 80% | 78.1% (74.6–81.2) | 0.51 (0.41–0.62) | 9.7 | 172× | 70.7% |
| DEEP | 85% | 83.9% (81.2–86.4) | 0.94 (0.77–1.10) | 11.0 | 152× | 76.0% |
| DEEP | 90% | 89.4% (87.4–91.4) | **1.81 (1.52–2.10)** | 12.7 | 132× | **81.0%** |
| SMALL | 80% | 75.3% (70.9–79.0) | 0.85 (0.71–1.00) | 9.5 | 176× | 68.2% |
| SMALL | 85% | 81.4% (77.9–84.5) | 1.50 (1.27–1.71) | 11.0 | 152× | 73.7% |
| SMALL | 90% | 87.1% (84.2–89.5) | **2.68 (2.27–3.04)** | 13.2 | 127× | **78.9%** |

Against the references: CFAR alone = 1,653 FA/img; **free gate alone ≈ 40 FA/img at 90% retention**.
So the CNN removes a further ~15–27× of the false alarms the gate leaves, at the same retention.
The val→test threshold drift (90% target → 87–90% on test) is small here; on the earlier 180-image
test split it was larger (90% → 85%) purely from having ~500 ships.

Exported weights, requant constants, thresholds and 256 golden vectors (integer logits, bit-exact
to the reference): `Results/cnn_weights_hw/full_{small,deep,xl}/`.

**Throughput (model, not RTL; `hwcost.py`, 128 lanes, 70% utilisation, 231 candidates/img, 800×800
frames):** SMALL ≈ 38 img/s (24.5 Mpx/s) · DEEP ≈ 10 img/s (6.3 Mpx/s) · XL ≈ 4.7 img/s (3.0 Mpx/s).
The Weibull prescreen alone streams 50 Mpx/s. The 297-lane ceiling scales these ≈ 2.3×. **This is
the accuracy/throughput Pareto the paper can report**; the model that "fits" is not unique.

## 6. Can it be improved further under the DE10 constraints?

**Recheck result: the hardware budget is not what limits accuracy; throughput is.** The largest
model here (XL) uses 66% of M10K and 52% of DSP; accuracy was still improving with capacity when
the data was large enough.

What the evidence shows:
- **On the 1,200-image subset, capacity looked saturated** (4× width: nothing; 4× MACs: −15–35%).
- **On full HRSID it is not.** More data gave DEEP ≈ −25% (float test FA@85 ≈ 1.4 → 1.0–1.1; the
  two test splits and epoch counts (60 vs 40) differ, so treat this as indicative) and SMALL nothing;
  a still larger model (XL, 4.15 M MACs) gave another ≈ −25% (1.0–1.1 → 0.8, same test split, 3 DEEP
  seeds agree to ±0.1). The earlier "saturation" was a small-data effect, not a hardware one.
- **Remaining error is partly information-limited.** (Analysis on a 1,200-image-subset run.) The
  10% of reachable ships lost at 90% retention are small and dim: largest-cluster median area 6 px
  (kept ships: 17 px); 16% have a 1-px largest fragment (kept: 3%); their brightest pixel is no
  brighter than ~45 gated false alarms/img.

**Not tested / not evidence:** a 14.5 M-MAC ceiling teacher was started and **aborted** (≈ 3 min/epoch
under GPU contention). Models > 4.15 M MACs, ensembling, distillation and FOV > 64 px were not tried.

**Where headroom plausibly remains:**
1. **More capacity**, paid for in frames/second (XL → 4.7 img/s at 128 lanes; ≈ 11 img/s at 297 lanes).
2. **Prescreen operating point** (Pfa, SLI, gate threshold): moves the reachable-ship rate (now
   90.6%) and the candidate load jointly; not swept for the cascade.
3. **Field of view beyond 64 px** (needs re-extraction at 96/128 px; costs patch-buffer M10K).
4. Distillation/ensembling (not tried).
5. **Generalisation:** HRSID only. SSDD (JPEG-compressed) was not evaluated; cross-dataset
   behaviour is unknown and the weights are tuned to HRSID's sensor/resolution mix.

## 7. Open hardware items this study surfaces

- Patch row buffer (64 rows × 8-bit × W) and a delayed detection plane for the trigger are **new
  RTL**, not reuse of the Weibull window.
- At ~1,670 raw candidates/image the trigger will burst; a patch FIFO costs M10K
  (64×64×8 b ≈ 4 M10K per patch). Put the gate *before* the FIFO (recommended), or the source
  must be back-pressurable (fine for the ROM-driven demo, not for live streaming).
- The gate (3×3 mean − c1) is computed at the trigger pixel from `fe.x` and `c1` — both already
  exist in `front_end3`.
- Throughput and the CNN-side resource figures are a model; they need an RTL MAC-array build and a
  Quartus fit with the CNN co-resident to become paper numbers. Only the Weibull row in §2 is measured.
- Weibull at W=800 compiled end-to-end (`_quartus/weibull_w800/`): fit successful, **timing closes at
  50 MHz** — setup slack +2.254 ns (Slow 1100mV 85C) / +2.026 ns (0C), hold +0.152 / +0.014 ns, TNS 0.
  Hold margin at 0C is thin (+0.014 ns) but positive; re-check once the CNN is co-resident.

## 8. Reproduce

```
matlab: extract_cnn_patches_hw('NumImages',5604,'PatchSize',64,'Pool',2,'GateTau',0.40,'OutName','cnn_patches_hw_full32.mat')
matlab: count_cfar_clusters()                                   # true CFAR-alone candidate count per image
HWDATA=full32 HWGATE=0.45  python cnn/train_hw.py --frag 2 --neg-ratio 16 --epochs 40 --size 32 --jitter 2 \
        --convs 5-16-1,3-32-0,3-32-1 --fcs 64 --name f_deep_s1            # DEEP; SMALL/XL: see cnn/run_full.sh
HWDATA=full32 HWGATE=0.45  python cnn/quant_hw.py qat Results/hw/f_deep_s1.pt --jitter 2 --export Results/cnn_weights_hw/full_deep
MODELS='{"XL":"f_xl_s1","DEEP":"f_deep_s1","SMALL":"f_small_s1"}' python cnn/final_report_hw.py
```
Ablations (§4): `HWDATA=64 HWGATE=0.45`, `cnn/run_study*.sh`, `cnn/compare_runs.py`.
Code: `cnn/hwlib.py` (data, metrics, nets), `cnn/train_hw.py`, `cnn/quant_hw.py`, `cnn/hwcost.py`,
`cnn/final_report_hw.py`; extraction `extract_cnn_patches_hw.m`, `count_cfar_clusters.m`.
Quartus W=800 probe: `_quartus/weibull_w800/`.

## 9. RTL cascade, hardware-faithful pipeline, and hardware-exact end-to-end numbers (added later 2026-10-02)

Everything in §5 used an idealised front end (3x3-mean gate, centroid-centred patches). The RTL cannot do that
at stream rate, so the training data was **rebuilt to the exact hardware spec** and the models retrained.

**Hardware spec** (`rtl/cascade/README.md`; training data `extract_cnn_patches_hwspec.m`): every pixel -> 256-entry
quantiser ROM -> 2x2-pooled frame store (1/4 size; patches are cut from it, no full-res frame is kept) -> Weibull
detection -> streaming NMS trigger (`D & ~left & ~up-left & ~up & ~up-right`, needs one row of detect bits) ->
gate `x - c1 >= tau` (both exist at the trigger pixel) -> 32x32 edge-replicated window -> INT8 CNN. The CNN runs
*after* the frame has streamed (it is 20-100x slower than the pixel stream), pulling windows from the pooled store.

**RTL** (all new, `rtl/cnn/`, `rtl/cascade/`; the verified Weibull RTL is untouched):

| | result |
|---|---|
| `cnn_core` vs Python integer reference | **bit-exact**: SMALL 32/32 and 24/24 (two models), DEEP 6/6 golden vectors |
| `cascade_top` system test (128x128 HRSID crop) | pooled store exact; trigger+gate events 37/37 and 24/24 match an independent recomputation from the RTL's own detect/x/c1 stream; every CNN logit bit-exact |
| Quartus, **800x800**, DEEP (`_quartus/cascade_deep/`) | **21,866 ALM (52%), 428 M10K (77%), 47 DSP (42%)**; setup +2.10 ns (85C) / +1.99 ns (0C), hold +0.18 / +0.08 ns at 50 MHz |
| `cnn_core` (serial) alone, DEEP | 1,562 ALM, 147 M10K, 33 DSP, Fmax 93.7 MHz |
| `cnn_core_q4` (quad) alone, DEEP | 8,007 ALM, 148 M10K, 65 DSP, Fmax 113.5 MHz |
| board demo (`_quartus/cascade_de10/`, 128x128 on-chip crop) | `.sof` built and **run on the DE10-Standard 2026-10-03**: default switches gave 24 candidates / 7 accepted, matching simulation; see `rtl/cascade/BOARD_BRINGUP.md` |

Cycle counts are measured in RTL simulation (not modelled). The first `cnn_core` (32 lanes, one output pixel at a time) took
**40.1 k cycles/patch (SMALL), 71.0 k (DEEP)** → 4.2 / 2.5 frames/s at 800×800 with ~274 gated events per image. That was the
weak point (the modelled 10 / 38 img/s of §5 had assumed 128 lanes), so it was fixed in a follow-up (below).

**Throughput upgrade (same day, `rtl/cnn/cnn_core_q4.v`, `rtl/cascade/cascade_top_2clk.v`).** Two changes, both bit-exact:
a quad-pixel core (2×2 output pixels per pass from one weight word, feature maps in 4 parity banks, pooling on raw accumulators)
and a 100 MHz PLL clock for the CNN stage with the Weibull stream staying on 50 MHz (dual-clock RAMs + 4-phase handshake).

| | serial @50 MHz | quad @50 MHz | **quad @100 MHz** |
|---|---|---|---|
| cycles/patch SMALL / DEEP / XL (RTL sim) | 40.1k / 71.0k / (172k est.) | 10.4k / 19.7k / 48.2k | same, 2× clock |
| frames/s at 800×800 (274 events/frame): SMALL | 4.2 | 13.2 | **22.5** |
| DEEP | 2.5 | 7.9 | **14.4** |
| XL | 1.0 | 3.5 | **6.8** |

The frame-time model (`rtl/cascade/throughput.py`) reproduces the simulated two-clock frame to 0.02% (154,516 vs 154,491 cycles), so
these are measured-cycle numbers, not guesses. Quad core: bit-exact on SMALL 64/64, DEEP 48/48, XL 12/12 golden vectors; the two-clock
cascade passes the independent store/event/logit check and a second frame through the same hardware is identical to the first.
Quartus at 800×800 with two clocks: DEEP 27.4k ALM (65%), 428 M10K (77%), 79 DSP (71%), setup +2.94 ns (50 MHz) / +1.40 ns (100 MHz);
**XL fits too but at 553/553 M10K (100%, no margin)** (+1.07 ns at 100 MHz). The quad core alone has Fmax 113 MHz. Remaining headroom:
overlap patch fetch with the previous inference (~5%), more lanes/pixels, a faster CNN clock (core Fmax 113 MHz).

**Retrained on the hardware-faithful data, evaluated on bit-exact fixed-point events.** Thresholds are chosen on
the float-pipeline *validation* split and applied to (a) the float-pipeline test split and (b) **hardware-exact
test events** (bit-exact fixed-point Weibull detection, trigger, hardware gate, pooled patches; same 842 test
images). The two agree, so the float-derived training/validation data is representative of what the RTL sees.

| Model (INT8) | Val target | HW-exact test ship retention (95% CI) | FA/img (95% CI) | Candidates/img out |
|---|---|---|---|---|
| XL (147k params) | 90% | 88.0% (85.3-90.2) | **1.74 (1.44-2.08)** | 15.2 |
| DEEP (66k) | 90% | 87.9% (85.2-90.0) | **1.98 (1.65-2.34)** | 15.2 |
| SMALL (16k) | 90% | 86.8% (84.2-89.0) | **2.71 (2.32-3.21)** | 15.7 |
| DEEP | 85% | 81.4% (78.3-84.6) | 1.02 (0.83-1.24) | 13.1 |
| DEEP | 80% | 76.5% (72.7-80.2) | 0.55 (0.42-0.73) | 11.7 |

Hardware-exact test: 842 images, 2,877 GT ships, **2,570 CFAR-reachable (89.3%)**, 1,630 triggers/img, 274 gated
events/img. So the funnel is **1,630 candidate triggers -> ~15 candidates/img (~107x fewer) at ~88% ship
retention; end-to-end ship recall ~78.5% of all GT ships**. Versus the idealised pipeline of §5 (DEEP: 89.4% at 1.81)
the hardware-faithful front end costs about +10% false alarms and ~1.5 points of retention — the price of a
trigger/gate/pooling that a streaming datapath can implement. (Full table: `Results/hw/eval_hwexact.json`,
`cnn/eval_hwexact.py`.)

**A bug this work found and fixed:** the first board wrapper never drove the last pixel of the frame
(it overwrote `pixel_in_valid` in the same cycle), so the frame never completed; found because the wrapper
simulation hung with the controller waiting for the frame-complete flag. The older per-detector board wrappers
(`*_de10_top.v`) use the same sequencer pattern and should be checked for it.

## 10. Board sweep over 250 images (JTAG-fed cascade) and a layout bug it found (2026-10-03)

**What was built.** `rtl/cascade/jtag/` + `_quartus/cascade_jtag/`: the same two-clock cascade behind a JTAG-to-Avalon master (Qsys), so the PC can push any
800x800 image and read every candidate event + integer CNN logit back, no HPS/Linux/SD needed. The Weibull line buffer needs a gap-free stream and a
full frame (5.1 Mbit) does not fit on-chip next to the cascade, so the stream-domain clock is gated (`altclkctrl`): it only ticks on cycles that carry a
pixel and the core sees an ideal stream at whatever rate JTAG delivers (~2.5 MB/s, 0.25 s per frame). Simulation (`jtag/tb`, random-gap feed, 2 frames)
passes `tb/check_cascade.py`. Fit @800x800 DEEP: 27.2k ALM (65%), 487 M10K (88%), 79 DSP; setup +1.51 ns (100 MHz) / +2.90 ns (50 MHz), hold >= +0.22 ns.
Event/result RAMs hold 8192 events per frame: with the 1e-3 plane at tau 0.75, 17 of the 250 images exceed the original 1024 (worst 4153; the
hardware flagged it, `ev_overflow`).

**Sweep.** `_comparison/cnn/sweep250/` (README there): 250 held-out test images x 4 Pfa planes = 1000 frames run on the FPGA; theta set to the minimum so
every gated event's logit is returned, accept decisions applied offline. Pd / pixel-Pfa use one definition for both systems (see `sweep250.py`).
**Result:** the 178,769 hardware logits equal the Python integer model's logits on every event (0 mismatches), and the hardware event lists equal the
model's trigger+gate lists on all 1000 frames. DEEP @ plane 1e-3: Weibull-only Pd 0.904 / pixel Pfa 4.5e-3 / 1612 false events per image;
cascade at the 80 / 85 / 90 % CNN points: Pd 0.674 / 0.724 / 0.803, Pfa 5.1e-6 / 9.7e-6 / 2.0e-5, 0.41 / 0.82 / 1.73 false events per image.
(Weibull-only values are the bit-exact fixed-point model; the pixel map itself is not read out of the FPGA. They agree with the earlier HPS hardware
measurement, 0.901 / 4.1e-3 on 200 images.) Board timing: the CNN pass costs ~0.2 ms per candidate event.

**BUG FOUND AND FIXED: the deployed CNN saw transposed windows.** The network was trained on arrays read by h5py from MATLAB v7.3 files, which store
pat(r,c,n) as [n][c][r], i.e. the pooled-store window transposed; `patch_fetch.v` streams it row-major. cnn_core was verified bit-exact on golden vectors
in the TRAINING layout and the cascade against a reference that cut the window row-major, so neither check could see the mismatch; the first real frames
did (hardware logits equalled the model's only for the transposed input). Effect of the mismatch on the 250-image sweep (as built): Pd 0.690 / 0.744 /
0.793 instead of 0.674 / 0.724 / 0.803 at the three operating points, false events 0.48 / 0.93 / 1.87 instead of 0.41 / 0.82 / 1.73 (small, mixed sign,
so the CNN is roughly transpose-tolerant, but it was not the validated function). **Fix (no retraining, no RTL change):** `_comparison/cnn/transpose_export.py`
transposes every conv kernel and re-indexes the first FC (exact: `int_forward(ints_T, P) == int_forward(ints, P^T)` bit for bit, proven on the 256 golden
vectors + 512 random patches); new tables `rtl/cnn/genq_hw_{small,deep,xl}_T/`; `run_q4_golden_T.sh` PASS 64/64, 48/48, 12/12 at unchanged cycle counts.
All board designs now use `genq_hw_deep_T`. Everything in sections 5-9 that describes accuracy refers to the training-layout function, which the corrected
RTL now reproduces exactly (verified on the board by the 0-mismatch result above); the pre-fix bitstream numbers are kept in
`Results/sweep250/*_prefix_transposed_bug*`. Quartus 21.1 hit an internal error (`opt_arm_carry`) on the transposed DEEP tables; setting
`SYNTH_TIMING_DRIVEN_SYNTHESIS OFF` in the qsf avoids it (`cascade_jtag`, `cascade_de10`).
