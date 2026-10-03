# Paper 2 — fixed-point model and RTL of the pooled Weibull prescreen (2026-10-03)

Code: `_comparison/fixedpoint/` (golden model, evaluation), `rtl/prescreen/` (RTL, testbenches, README), `_quartus/prescreen_probe/` (fit/timing probe).
Results: `_comparison/Results/fixedpoint/` (`eval.json`, `fx_vs_float_pd.csv`, `fx_decision_agreement.csv`).

## 1. Design decision: two-pass cascade, integer arithmetic

The streaming Weibull core of the earlier cascade needs a gap-free pixel stream, leaves the outer 8 px unevaluated and works on the full-resolution 17/13 window. The software study (roadmap §11) showed that the
winning prescreen is *pooled 2×2, window 25/17, Pfa 3e-2, gate 0.6, 5×5 contrast-peak events, padded border* (config A). That needs the padded border and a 2×2 pooled image, both awkward in a stream, so:

* pass 1 (50 MHz): QROM → 2×2 pool → pooled store (400×400×8 bit; this store already existed for the CNN patches);
* pass 2 (100 MHz): the prescreen reads the stored pooled frame back with mirrored addresses (border padded by 12 px, edge sample repeated) and emits events;
* pass 3: patch fetch → CNN, unchanged.

The prescreen works on the 8-bit pooled code `P` itself, so every quantity is an exact integer:

`S1 = Σ P`, `S2 = Σ P²` over the ring (N = 25² − 17² = 336 cells); `A = N·P − S1` (contrast × N); `num = N·S2 − S1² = N(N−1)·c₂`.
The Weibull rule `x > c₁ + K/C`, `C = clamp(π/√(6c₂), 0.8, 8)`, `K = γ + ln(−ln Pfa)` is the same inequality as
`A > 0 and A² > κ²·N/(N−1)·clamp(num, NUM_LO, NUM_HI)` with `κ = K√6/π`; the shape clamp turns into a clamp on `num`. The product is evaluated as `A² > (clamp(num) >> 12) · KC`, `KC = round(κ²N/(N−1)·2¹²)`
(Pfa 3e-2 / 1e-2 / 1e-3 / 1e-4: 8381 / 11060 / 15733 / 19546). No ROM, no square root, no divider. Gate: `A ≥ G`, `G = round(N·τ/step)` = 16065 for τ = 0.6.

Bit widths are the analytical worst cases for N = 336 and 8-bit data: `P` 8 u; column sums 13 u (x) / 21 u (x²); horizontal sums 18 u / 26 u; `S1` 17 u (≤ 85,680); `S2` 25 u (≤ 21.85 M); `A` 18 s; `N·S2`, `S1²`, `num`, `A²` ≤ 7.3·10⁹ → 34 u; the clamped `num >> 12` is 19 u and `KC` 15 u.

## 2. What the arithmetic costs in detection (all 5,604 images, 16,951 ships; strict event Pd = a gated event within 4 px of the ship polygon)

| config | float (exact logs) | 8-bit data, ideal arithmetic | **integer (RTL model)** | events/img (integer) |
|---|---|---|---|---|
| **A: pooled 25/17, Pfa 3e-2, τ 0.6** | 0.9959 | 0.9961 | **0.9961** (66 missed) | 181 (median 33, max 2,616) |
| pooled 17/13, Pfa 1e-2, τ 0.75 | 0.9854 | 0.9859 | 0.9859 | 91 |
| pooled 17/13, Pfa 1e-3, τ 0.75 | 0.9721 | 0.9727 | 0.9727 | 83 |
| pooled 17/13, Pfa 1e-4, τ 0.75 | 0.9466 | 0.9473 | 0.9473 | 73 |

* The integer arithmetic (fixed-point constants, truncated `num`, clamp) changes 5.6 parts per million of the pixel decisions in config A (17–46 ppm in the others) relative to ideal arithmetic on the same 8-bit data; the event Pd is identical in all four configurations, and the event sets differ in 0 (A) to 53 events over the whole data set.
* Quantising the data to 8 bits changes 3.7–5 % of the *detected* pixels (borderline pixels) but not Pd (+0.0002 to +0.0014, within noise); the extra events are ties of the integer contrast.
* Test split (842 images, 2,877 ships): config A 0.9972 integer vs 0.9962 float.
* The float figures reproduce the MATLAB sweep (`pd_sweep3.m`: 0.9847 / 0.9712 / 0.9451; the small differences come from the Python polygon rasteriser).
* Event load is very skewed: the median image has 33 events, 65 images exceed 2,048 and the maximum is 2,616, so the 8,192-entry event RAM cannot overflow; CNN time is dominated by clutter-rich scenes.

## 3. RTL

`prescreen_core.v` (sequencer with mirrored addressing, running column sums, decision), `peak5.v` (5×5 contrast peaks), `prescreen_top.v`, `cascade_ps_2clk.v` (whole two-clock cascade), `jtag/cascade_ps_jtag_{core,top}.v` (no clock gating needed).
Five store reads per padded pixel (one per clock) → (400+24)²·5 = 0.90 M clk_cnn cycles = 9.0 ms per frame, independent of content (a single-read-port design; a line-buffer variant would remove most of it at the cost of ~14–25 M10K).

Verification (iverilog; every comparison is exact, not statistical):
* `tb/run_regression.sh`: prescreen vs the Python golden model on six crops (real HRSID scenes, one synthetic), window 25/17, 21/15 and 17/13, all four Pfa planes, non-square sizes, each run twice back to back — **6/6 bit-exact** (events j, i, A, S1, num>>12, P).
* `tb/cascade_ps_tb.v` + `check_cascade_ps.py`: pooled store, events and every CNN logit (existing DEEP INT8 tables) vs Python, second frame identical:
  128×128 crop (8 events) and 192×192 crop with an irregular, gappy pixel feed (52 events) — **all PASS**.

Quartus 21.1 (`_quartus/prescreen_probe`, 800×800 incl. the 160 kB store, Cyclone V 5CSXFC6D6F31C6, 100 MHz constraint): **826 ALMs (2 %), 8 DSP, 181 M10K (125 of them the store)**, setup slack **+0.98 ns (85 °C) / +0.88 ns (0 °C)**, hold +0.22 ns.
Timing history: first version −5.1 ns (event-peak decision after the RAM outputs), −0.57 ns after one pipelining step, met after the second.

## 4. CNN on the hardware-exact candidates, INT8, and the whole cascade in RTL

* Candidate set (`extract_fx_events.py`): 1,015,171 events from all 5,604 images (181 per image, 8.4 % on ships), patches cut row-major from the pooled store exactly as `patch_fetch.v` streams them (no transposition fix needed, unlike the older MATLAB-extracted sets).
* Float ablation on this set (test split, cascade Pd at the 97 % validation-retention target): plain network + ship-level loss + hard negatives 95.4 % @ 3.11 false events/img; + side features 95.1 % @ 2.39; + context tower + side features 95.3 % @ 1.90.
* The plain network maps onto the existing CNN core unchanged, so it was taken to INT8 first (`cnn/train_q.py`, `export_q8.py`): post-training quantisation alone gives the float result (val 2.85 vs 2.83 FA/img at 97 %); after QAT the INT8 cascade is **95.4 % @ 3.24 false events/img** on the test split (float 3.11). The integer model used for scoring is bit-exact with the RTL arithmetic (GPU integer model vs `quant_hw.int_forward`: max diff 0.0).
* RTL tables `rtl/cnn/genq_fx_plain/`: CNN core golden regression **64/64 vectors bit-exact**, 19,658 clk per patch. Thresholds from validation: theta = -14607 / -17457 / -20129 for 95 / 97 / 98 % retention.
* Whole cascade (`cascade_ps_2clk`, INT8 tables): store, events and every logit/accept flag bit-exact vs Python on a 192 x 192 crop with a gappy pixel feed (54 events) and on the JTAG wrapper (`cascade_ps_jtag_core`, register-level flow with a random-gap host, 8 events, second frame identical).
* Quartus 21.1 of the complete JTAG-fed cascade (`_quartus/cascade_ps_jtag`, 800 x 800, Cyclone V 5CSXFC6D6F31C6): **11,172 ALMs (27 %), 397 M10K (72 %), 74 DSP (66 %)**; all setup / hold slacks positive (clk_cnn 100 MHz setup +1.49 ns slow-85 C, CLOCK_50 +3.98 ns, minimum hold +0.10 ns). The earlier streaming-Weibull design used about 65 % of the ALMs and 88 % of the M10K blocks.
* **Board (DE10-Standard, programmed 2026-10-03 23:52, JTAG-fed, pfa plane 0, theta -17457): 250 frames / 22,790 events and 1000 frames / 71,380 events, every frame bit-exact** (events j, i, logits and accept flags vs `prescreen_fx` + `quant_hw.int_forward`; no event-RAM overflow, no dropped pixels). Prescreen pass 9.01 ms on every frame (900,513 clk_cnn cycles; matches the RTL simulation of a full 800 x 800 frame, also bit-exact). Host: `sweep250/board/board_ps.py`, `board_sweep_ps.py`; results `Results/sweep_ps250`, `sweep_ps1000`.

## 5. Context-tower cascade (2026-10-04): INT8, RTL, board

Network (`cnn/train_q3.py`, tied-scale QAT): fine tower (as before, 800 features) + **context tower** (32 x 32 patch of P4 = 2x2 pool of the pooled frame = 128 x 128 original px; conv5 1->8 +pool, conv3 8->16 +pool, conv3 16->16 +pool -> 64 features) + **side features** (9 hardware-exact uint8 codes, `fixedpoint/side_fx.py`: event contrast, local ring mean and sigma, event amplitude; image mean, s.d., bright and dark pixel fractions, ln(1+events)) -> fc 16, head fc (800+64+16) -> 64 -> 1. The three feature segments share one activation scale so their integer partial sums add inside one accumulator.

* Float on hardware-exact codes: 95.9 % @ 2.39 false events/img (97 % target, test split); INT8 after QAT (PTQ already equal): **96.2 % @ 2.64, 93.6 % @ 1.28 (95 % target)**; single-tower INT8 gives 95.4 % @ 3.24 and 93.4 % @ 2.07, i.e. about +2 points of recall at equal false events and inshore recall unchanged at the 95 % target (88.7 %) and 91.8 -> 93.3 % at the 97 % target. Thresholds from validation: theta = -5979 / -10695 / -14462 / -17056 for 90 / 95 / 97 / 98 % retention.
* RTL (`rtl/cnn/cnn_core_ctx.v`, generator `gen_cnn_rtl_ctx.py`): cnn_core_q4 generalised to three input streams (`in_kind` fine / context / side, any order), per-layer source / destination memory and base offsets, and layers whose accumulation runs over up to three input segments (the concatenated head). A 9-layer program; bank depth 1,184 (still one M10K per bank memory). **48/48 golden vectors bit-exact (6 input orders), 28,003 clk per candidate** (single tower 19,658).
* New blocks (`rtl/ctx/`): `ctx_fetch.v` (context patch = 4,096 store reads, bit-exact incl. corners), `side_unit.v` + `isqrt_pipe.v` (per-event codes, bit-exact on 3,000 random inputs), `img_stats.v` / `img_codes.v` (image statistics accumulated in pass 1, f4..f7 once per frame; bit-exact on 60 real images + edge cases), `ln_rom.v`, `cascade_ctx_2clk.v` (top; event RAM 4,096 x 52 bit), `jtag/cascade_ctx_jtag_{core,top}.v` (ID 0xCA5CADE4).
* System simulation (192 x 192 crop, 54 events, gappy feed): store, events + side codes, image codes and every logit exact (`rtl/ctx/check_cascade_ctx.py`).
* Quartus (`_quartus/cascade_ctx_jtag`): **11,856 ALM (28 %), 394 M10K (71 %), 96 DSP (86 %)**, setup +0.78 ns at 100 MHz, +3.97 ns at 50 MHz, all hold slacks positive.
* **Board (DE10-Standard): 5 frames, then 1,000 frames / 71,380 events, every frame bit-exact** (events, logits, accept flags vs the Python models). After-last-pixel time (prescreen + CNN) mean 32.7 ms, median 16.6 ms, p95 94.6 ms, max 755 ms. Host: `board_ctx.py`, `board_sweep_ctx.py`; results `Results/sweep_ctx1000`.

## 6. Whole-data-set board sweep and training statistics (2026-10-04)

**Whole data set on the board** (`board_sweep_ctx.py --n 5604`, context bitstream, theta -14462 stored with every event's logit; `Results/sweep_ctx_all`): 5,604 frames, 1,015,171 events, 0 mismatches against the golden models, prescreen 9.01 ms on every frame. `fixedpoint/eval_full_sweep.py sweep_ctx_all ctx` evaluates two systems from this one sweep: Weibull only (every event accepted) and Weibull + CNN (logit >= theta, thetas -5979 / -10695 / -14462 / -17056 from validation for 90 / 95 / 97 / 98 % retention), with 1,000-resample image bootstraps (`Results/fixedpoint/full_ctx_metrics.csv`, `full_ctx_curves.json`, figure `Figures/fixedpoint_full_sweep.png`, `fixedpoint/fig_full_sweep.py`).

| subset | system | recall (95 % CI) | inshore | offshore | false events / img | precision |
|---|---|---|---|---|---|---|
| all 5,604 | Weibull only | 99.61 % (99.48-99.73) | 99.22 | 99.98 | 166.0 | 0.018 |
| all | + CNN 90 % | 88.57 | 77.98 | 98.50 | 0.38 | 0.876 |
| all | + CNN 95 % | 94.08 | 88.52 | 99.30 | 1.12 | 0.717 |
| all | + CNN 97 % | 96.70 (96.32-97.04) | 93.59 | 99.61 | 2.31 | 0.558 |
| all | + CNN 98 % | 97.67 | 95.49 | 99.71 | 3.62 | 0.450 |
| test 842 | Weibull only | 99.72 (99.52-99.89) | 99.52 | 99.93 | 172.6 | 0.019 |
| test | + CNN 95 % | 93.60 (92.04-95.04) | 88.74 | 98.65 | 1.28 | 0.714 |
| test | + CNN 97 % | 96.21 (95.07-97.19) | 93.38 | 99.15 | 2.64 | 0.555 |
| test | + CNN 98 % | 97.43 (96.61-98.15) | 95.57 | 99.36 | 4.05 | 0.451 |

Train, validation and test recalls agree within their intervals at every operating point (train and validation are in-sample for the CNN; the prescreen has no training). The CNN removes 98.6 % of the false events at the 97 % target for a 2.9-point recall cost on all images. The test rows reproduce the software evaluation (96.2 % @ 2.64).

**Seeds and splits** (`cnn/run_seeds.sh`; `fixedpoint/seeds_metrics.py`, `seeds_report.py`, `seeds_pair.py`; `Results/fixedpoint/seeds_all_runs.csv`, `seeds_summary.csv`, `seeds_paired.csv`): per network five runs = original (sd1) + seeds 2, 3 on the original split + splits 20261004 and 20261005 (seed 1), each with its own validation thresholds and test images; test-split mean +- s.d. over the five runs:

| target | context recall | context FA / img | single-tower recall | single-tower FA / img |
|---|---|---|---|---|
| 90 % | 88.2 +- 1.5 | 0.52 +- 0.12 | 87.8 +- 1.8 | 0.79 +- 0.13 |
| 95 % | 93.9 +- 1.1 | 1.47 +- 0.38 | 93.7 +- 1.3 | 2.14 +- 0.33 |
| 97 % | 96.0 +- 0.8 | 2.54 +- 0.58 | 95.9 +- 1.1 | 3.40 +- 0.59 |
| 98 % | 97.3 +- 0.8 | 3.83 +- 0.84 | 97.2 +- 0.9 | 5.18 +- 0.83 |

Paired by run, the context network has fewer false events in 5 of 5 pairs at every target (mean reduction 35 / 32 / 26 / 27 % at 90 / 95 / 97 / 98 %), while the recall difference is +0.1 to +0.4 points on average with a range that includes zero and negative values. **Correction of the earlier claim**: the "+2 points of recall at equal false events" came from the single original run; across runs the robust effect of the context tower is fewer false events at about equal recall. Run-to-run recall s.d. (about 1 point) is larger than the recall difference.

## 7. Open items

* Power measurement (not done for any design).
* Optional: overlap the next candidate's fetch with the CNN compute (the core input memory is currently idle-blocked for about 15 % of each candidate's time).
* Optional: on-board whole-data-set sweep of the single-tower bitstream (not needed for the comparison; its software scores cover all images).
