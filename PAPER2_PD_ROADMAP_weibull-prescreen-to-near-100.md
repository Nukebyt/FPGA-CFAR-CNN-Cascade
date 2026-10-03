# Paper 2 — Roadmap: raise the Weibull prescreen Pd to ~100 % so the cascade stays above 95 %

Created 2026-10-03. Scope: **software / MATLAB first** (HRSID, bit-exact fixed-point Weibull model + Python INT8 CNN); FPGA only after the
algorithm works. Separate from Paper 3's hardware files (shares only read-only RTL/model code).

## 0. Where we are (measured, not assumed)

| | value | source |
|---|---|---|
| Weibull prescreen Pd, SLI 17 / guard 13, Pfa plane 1e-3 | **0.904** (250 test images, 696 ships) ; ~89.3 % of the 2,877 test ships are "CFAR-reachable" | `Results/sweep250`, study §3 |
| Pd at the other planes 1e-4 / 1e-5 / 1e-6 | 0.789 / 0.674 / 0.573 | same |
| Weibull false events per image at 1e-3 (NMS triggers, gated) | ~1,612 (≈274 after the x−c1 ≥ 0.75 gate) | same |
| Cascade end-to-end Pd (CNN @ 80 / 85 / 90 % retention) | 0.674 / 0.724 / 0.803 at 0.4 / 0.8 / 1.7 false events per image | board-verified |

The CNN can only remove candidates, never add ships, so **the cascade Pd is capped by the prescreen Pd times the CNN retention**
(0.904 × 0.9 ≈ 0.80). To finish above 95 % with a CNN that keeps ~97 % of the ships it is given, the prescreen has to reach ≳ 98–99 %.

**Success criteria (proposed, to be confirmed after phase 1):** (a) prescreen box-hit Pd ≥ 98 % (stretch 99.5 %) on held-out images; (b) integrated cascade Pd ≥ 95 %
with ≤ ~5 false events per image (current CNN workload budget ≈ 15 candidates/image); (c) every gain measured with bootstrap CIs on images never used for tuning.
**Caveat decided up front:** HRSID contains ships that are barely visible or whose annotation is generous; a 100 % ceiling may not exist at any sane false-alarm
budget. Phase 1 therefore measures the *oracle ceiling* (best Pd reachable at Pfa = 1e-1 with the float model, and how many annotated ships are below the sea clutter itself) before we chase it.

## 1. What the literature and our research vault say

Light pass (vault search + scholar APIs, 2026-10-03). The full `/hyperresearch` pipeline has not been run for this topic; an adversarial pass can be added if the paper claims depend on it.

| Idea | Evidence found (verified record) | Use here |
|---|---|---|
| CFAR window contamination by neighbouring targets / clutter edges ("masking") is a classic cause of missed and false detections | Zaimbashi, *adaptive CA-CFAR for interfering targets and clutter-edge situations*, Digital Signal Process. 31, 2014 (vault note); El-Darymli et al., *Target detection in SAR imagery: a state-of-the-art survey*, J. Appl. Remote Sens. 7, 2013, 10.1117/1.jrs.7.071598 | Phase 1 measures how many misses are "window inflation" (c1 above the true sea level) vs. true low contrast |
| Censoring: remove bright reference cells before estimating clutter (iterative / bilateral censoring) | *Improved iterative censoring scheme for CFAR ship detection*, IEEE TGRS 2013, 10.1109/tgrs.2013.2282820; *Approximate MLE based automatic bilateral censoring CFAR…*, DSP 2023, 10.1016/j.dsp.2023.103972; Gaofen-3 chain (ICS-CFAR + CNN), Sensors 18(2):334, 2018 (vault) | Phase 4 variant C |
| Window-free / region-based CFAR (superpixels) | *Superpixel-level CFAR detectors for ship detection in SAR imagery*, IEEE GRSL 2018, 10.1109/lgrs.2018.2838263; *Fast superpixel-based non-window CFAR…*, Remote Sens. 14(9), 2022, 10.3390/rs14092092 | Phase 4 variant (software-only reference) |
| Two-parameter / log-normal / robust (MAD) CFAR | RadarConf 2017 *two parameter CFAR in Log-Normal clutter*, 10.1109/radar.2017.7944196; *CFAR … median absolute deviation thresholding*, SIVP 2023, 10.1007/s11760-023-02513-2 | baselines for the Pd–Pfa comparison |
| CFAR prescreen + CNN discriminator is an established architecture (so the Pd ceiling of the prescreen is the known weak point) | Gaofen-3 paper above; *Study on the combined application of CFAR and deep learning in ship detection*, J. Indian Soc. Remote Sens. 2018, 10.1007/s12524-018-0787-x; CFAR as *input to* a detector (different pattern): Remote Sens. 16(5):733, 2024 | framing |
| Dataset facts: 5,604 images / 16,951 ships, 800×800, resolutions 0.5 / 1 / 3 m (per the HRSID paper; per-image resolution is **not** in the annotations), inshore/offshore split lists, per-ship polygons | HRSID, IEEE Access 2020, 10.1109/access.2020.3005861 | segregation factors (scene, size); resolution only via proxies |
| Small ships are the hard class | LS-SSDD-v1.0, Remote Sens. 12(18), 2020, 10.3390/rs12182997 | expectation: misses concentrate in small / low-contrast ships |
| SAR clutter statistics (why a Weibull/log-based CFAR over- or under-shoots) | *Statistical modeling of SAR images: a survey*, Sensors 10(1), 2010, 10.3390/s100100775 | clutter-factor analysis |

Classic references we rely on but have **not** verified in the vault yet (look up before citing): OS-CFAR (Rohling 1983), VI-CFAR (Smith & Varshney 2000), Lee / Frost speckle filters, hysteresis (double-threshold) detection.
Not found in the pass: any prior work that measures, ship by ship, the exact decomposition of a CFAR miss (below), nor a resolution-aware cluster-size consistency rule evaluated on HRSID — treat as open, to be checked adversarially before claiming novelty.

## 2. Phase 1 — find out *exactly* why 10 % are missed (segregate the dataset)  [started today]

**Instrument:** `_comparison/pd_study/pd_study_extract.m` runs the bit-exact fixed-point and the float Weibull model on all 5,604 images and records, **per annotated ship** (polygon mask, box fallback):
hit / margin (x − T, log units) at Pfa ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6}; and, at the best ship pixel for 1e-3, the **exact decomposition of the gap**

`T − x_best  =  (c1 − μ_bg)  +  δ(c2, Pfa)  +  (μ_bg − x_best)`
 window/mean inflation · Weibull threshold offset · true contrast deficit against the local sea

so every miss is attributed to *where the missing margin comes from* rather than guessed. Also per ship: size (mask px, box), distance to the image border / CFAR-valid region
(the 17×17 window cannot evaluate the outer 8 px), nearest neighbour distance, local background mean / std / heterogeneity, brightest-pixel amplitude. Per image: false pixels, NMS triggers
and gated triggers at every Pfa (CNN workload), clutter descriptors. HRSID inshore/offshore lists are merged afterwards.

**Segregation factors** (each reported as Pd by bin with bootstrap CI, plus a decision-tree / logistic attribution so correlated factors are separated):
scene (inshore / offshore) · ship size · contrast Δx and z-score · local clutter level and heterogeneity (sdBg, c2) · neighbour distance (masking) · border proximity ·
image-level clutter (bright / dark fractions, land proxy) · ship-vs-window geometry (ring thickness is only (17−13)/2 = 2 px).

**Outputs:** miss-reason table (border / below-sea-level / inflation-dominated / threshold-dominated / near-miss), recoverability curve (fraction of misses recovered at each Pfa), the oracle ceiling, workload cost of each Pfa.
**Gate G1:** we know which reasons account for ≥ 90 % of the misses and what fraction is recoverable by (i) threshold, (ii) window/estimation changes, (iii) only by enhancement / learning.

## 3. Phase 2 — cheap levers and ceilings (software)

1. **Pfa relaxation curve** of the prescreen alone (float model, Pfa up to 1e-1) with pixel Pfa and trigger/gated-trigger counts (what the CNN would have to digest). Hardware has only four planes today — any gain here is "just a LUT".
2. **Border handling:** ships within 8 px of the edge are structurally undetectable now (valid region). Measure the Pd lost; test replicate / mirror padding in the model.
3. **Gate re-tuning** (x − c1 ≥ τ currently 0.75 removes weak ships before the CNN): Pd vs τ with the CNN workload.
4. **Hysteresis / double threshold:** low threshold detections kept only if they connect to ≥ k pixels or to a high-threshold pixel (cheap, streaming-friendly).

## 4. Phase 3 — spatial-consistency (resolution-aware plausibility) filter

The idea: a real ship is an *extended* structure at the sensor's resolution; isolated single-pixel detections cannot be ships and may be discarded, which **lets us lower the threshold** (more Pd) without flooding the CNN.
Plan: (a) from the polygon masks measure the pixel-area distribution of real ships per size class (HRSID mixes 0.5 / 1 / 3 m, so use pixel area, not metres, unless a per-image resolution proxy is found);
(b) for each candidate compute the *connected-component size at several thresholds* (cluster growth profile) and use it as a feature, not a hard rule — **risk:** the missed small ships are exactly the ones that occupy few pixels, so a hard minimum-area rule would trade Pd for Pfa; tune on the train split only;
(c) 2×2 pooling (already in the pooled store) as a spatial integrator: detect on the pooled grid and compare Pd of small vs extended ships.

## 5. Phase 4 — pre-Weibull enhancement / CFAR variants (ordered by cost-effectiveness and hardware cost)

| id | variant | mechanism it targets (from phase 1) | hardware class |
|---|---|---|---|
| A | window geometry: SLI/guard sweep, multi-scale union | ring too thin (2 px) / ship larger than guard | more line buffers |
| B | pooled-domain or 2-level (full-res + pooled) detection | speckle averaging for extended ships | uses existing pooled store |
| C | censored / robust background (OS-like, bilateral, MAD) | window inflation by neighbours and clutter edges | sorting network / second pass |
| D | speckle pre-filter (boxcar, Lee, Frost, non-local) before the log | low-contrast ships limited by speckle | small line-buffer filter |
| E | local-contrast / top-hat / matched filter on the log image | contrast deficit vs. the local sea | filter bank |
| F | superpixel / region CFAR (software reference only) | edge of clutter, inshore | not hardware-friendly; upper-bound |
| G | learned front end: tiny CNN heat-map on a coarse grid in place of (or in union with) CFAR | what is still missed | CNN-class resources |

Every variant is judged on the **same** metrics (Pd box-hit, strict Pd = ≥ k pixels, pixel Pfa, triggers/gated per image). Union detectors (A ∪ B ∪ …) are evaluated as a *recall ceiling* before any single-path design is committed.

## 6. Phase 5 — integrated cascade model (still software)

Retrain / re-threshold the CNN on the *new* candidate distribution (more, harder negatives from the lower threshold); report cascade Pd at fixed FA budgets with CIs; compare against the 78.5 % end-to-end baseline.
**Protocol:** HRSID split 70/15/15 by image (seed fixed in `hwlib.py`); every tuning decision (filters, thresholds, gate, CNN operating point) on train/val only; final numbers on test (842 images) and, for the board, the 1,000-image sweep (842 test + 158 val). Weibull itself has no training, so the diagnostic phase may use all 5,604 images.
**Gate G5:** cascade Pd ≥ 95 % at ≤ 5 false events/image on test.

## 7. Phase 6 — FPGA (deferred, only after G5)

Map the winning front-end change to RTL, re-verify bit-exact against the model, run on the board via the JTAG-fed flow (`rtl/cascade/jtag/`), compare resources against the DE10 budget (current cascade: 65–67 % ALM, 88 % M10K, 71 % DSP with the JTAG shell).

## 8. Parallel track (running now): 1,000-image on-board sweep

`Results/sweep1000/` (842 test + 158 validation images × 4 Pfa planes, JTAG-fed cascade on the DE10) gives the baseline with tighter CIs; per-image logs via `cnn/sweep250/per_image.py` (`SWEEP=sweep1000`).

## 9. Status log

* 2026-10-03: roadmap written; phase-1 extractor built and smoke-tested (float and fixed-point hit flags agree on the test ships); full-dataset extraction launched; 1,000-image board sweep launched. Results of phase 1 are appended below as they land.

(Details of the method and the exact loss accounting: `PAPER2_PD_LOSS_ANALYSIS_2026-10-03.md`.)

## 10. Phase 1 results (2026-10-03, all 5,604 images / 16,951 ships) — `_comparison/pd_study/`, outputs in `Results/pd_study/`

**Two corrections to the baseline before anything else.**
1. *The "90 % Pd" is a lenient number.* The box-hit definition counts any detected pixel inside the GT rectangle, so at high Pfa a box "hits" on sea clutter alone (box Pd reaches 100 % at Pfa 0.1 with 10 % of all pixels flagged).
   Measured on the same ships: hardware model at 1e-3 — box hit **0.891**, a detected pixel on the ship polygon **0.846**, a *gated trigger on the ship* (the event the CNN must receive; `evMask`) **0.843**
   (float model: 0.906 / 0.860 / 0.858). Offshore 0.861, inshore 0.823 (test split 0.860). **The real prescreen ceiling for the cascade is ≈ 84–86 %, not 90 %.**
2. *Hardware loses ~1.4 points relative to the float model* (0.8597 vs 0.8456 pixel-on-ship). **Correction:** this is not fixed-point arithmetic — all 233 such ships lie within 8 px of the image edge: the float front end evaluates the border, the RTL never evaluates the outer 8 px.

**Where the 15.7 lost Pd points go** (hardware model, Pfa 1e-3, mutually exclusive; ordered rules in `classify()`):

| cause | ships | Pd points |
|---|---|---|
| **own ship inside the CFAR reference ring (self-masking)** — would be detected against a clean sea | 1,586 | **9.36** |
| gate / NMS: ship pixels detected but no gated trigger lands on the ship | 323 | 1.91 |
| neighbouring ship in the ring (masking) | 257 | 1.52 |
| RTL border (float finds it, RTL never evaluates the outer 8 px) | 233 | 1.37 |
| low contrast, near miss (clean-sea margin −0.15…0 nat) | 135 | 0.80 |
| bright clutter in the ring | 53 | 0.31 |
| low contrast, deep (invisible at 1e-3) | 69 | 0.41 |
| other (variance/shape mismatch) | 8 | 0.05 |

For **91 % of the lost ships the clean-sea margin is positive**: against ship-free sea statistics (c1, c2 from the surrounding sea) the ship *would* be detected. The loss is dominated by the estimator, not by ship visibility:
with SLI 17 / guard 13 the reference ring is only 2 px thick and any ship wider than ~13 px puts its own pixels into c1 and c2 (the 2×2-guard-sized ships are not the problem — large, bright ones are).
Truly invisible ships (below the clean-sea threshold) are only ~1.2 points. The statistical predictor of a loss (gradient boosting, AUC 0.93, held-out images) ranks ship contrast, own-ship / bright-clutter / neighbour contamination of the ring and border distance highest; scene and ship count matter little once those are known.
By factor (strict Pd): neighbours touching (<0.5 px gap) 0.66; within 4 px of the image edge 0.66 (the outer 8 px are not evaluated); contrast dx < 1.0 nat ≈ 0.4; sub-25-px ships 0.28; local clutter sd < 0.3 (very flat/dark scenes) 0.65.

**What is recoverable by just relaxing the threshold (float, gated event on the ship):** Pd 0.858 (1e-3) → 0.905 (3e-3) → **0.940 (1e-2)** → 0.933 (3e-2) → 0.838 (1e-1) — and the gated events per image stay ≈ 270–290 up to 1e-2 because the x − c1 ≥ 0.75 gate caps the workload. Above ~1e-2 the NMS trigger collapses (detections merge into big blobs, the corner trigger is no longer on the ship): *the event definition, not the threshold, becomes the limit*.

**Consequences for the plan (reordering of phases 2–4):**
1. Biggest lever = fix self-masking (≈ 9–11 points): window geometry (bigger guard/ring, multi-scale), detection on the 2×2-pooled image (the ship is half as large relative to the same window), or censoring of the ship's own pixels. Re-evaluate SLI/guard under the strict metric — the old choice (SLI 17 / guard 13) was tuned on box-hit Pd, which cannot see this loss.
2. Cheap, nearly free: Pfa plane 1e-2 + retrain the CNN on that candidate distribution (+8 points, workload unchanged); event definition robust at high Pfa (centroid / component-level instead of the NMS corner) so that 3e-2 pays off; gate τ re-tuned (1.9 points); border handling in the RTL (pad/mirror the outer 8 px, 1.4 points).
3. Border handling (ships within 8 px of the edge, Pd 0.66) and neighbour masking (touching ships) are second-order.
4. Spatial-consistency filter: keep as phase 3 but it now serves to *buy back false events* after the threshold/window changes, not to find ships.
5. Ceiling check: invisible ships ≈ 1.2 points → a 97–98 % prescreen is plausible; 100 % is not.

**1,000-image board sweep (parallel track), DEEP INT8 on the FPGA:** 842 test + 158 val images × 4 planes = 4,000 frames, 725,893 events, **0 logit mismatches vs the Python model**. At plane 1e-3 (box-hit, 3,392 ships): Weibull-only Pd 0.899 / pixel Pfa 4.5e-3 / 1,603 false events per image; cascade at the 80 / 85 / 90 % CNN points Pd 0.692 / 0.736 / 0.795 with 0.53 / 0.99 / 1.95 false events per image (95 % CIs in `Results/sweep1000/summary_hw.json`; figures `fig_*_hw.png`; per-image log via `SWEEP=sweep1000 python per_image.py`).
Note the box-hit vs strict gap above: the cascade Pd on this sweep is also box-hit based.

## 11. Phases 2–5 first results (2026-10-03, software, float models) — prescreen on the POOLED image + peak events + retrained CNN

**Harness.** `_comparison/pd_study/pd_variants_eval2.m` evaluates 20 front-end groups × 6 Pfa × 5 gates × 4 event types (2,400 variants) per image on the strict metric (an event within 4 full-resolution px of the ship polygon; touching ships share events), plus CNN workload (events/image) and off-ship events. `variants_report.py` aggregates. Variants: window (sli/guard) 17/13 … 49/33, 2×2 and 4×4 pooling (mean of intensity or mean of log amplitude = what the pooled store holds), padded border, events = NMS corner / contrast peak (3×3, 5×5 local maximum of x−c1) / per-component. (v1 `pd_variants_eval.m` also had a censored-background group: iterative censoring did **not** help on this metric — it lowers c1 and floods the event stream — and was dropped.)
Selection used 600 train+val images (1,644 ships); every number below is on the 842 untouched **test images (2,877 ships)** unless stated.

**Prescreen Pd (event on the ship) vs CNN workload — chosen on the tuning images, measured on test:**

| prescreen | Pd | events/img |
|---|---|---|
| RTL baseline (full-res 17/13, border not evaluated, Pfa 1e-3, gate 0.75, NMS) | 0.867 | 274 |
| + padded border (same everything else) | 0.882 | 284 |
| full-res, bigger window 49/33 (padded), Pfa 1e-3, NMS | 0.931 | 333 |
| **pooled 2×2 log-mean, 25/17 window, Pfa 0.03, gate 0.6, peak 5×5 (config A)** | **0.9986** | **190** |
| pooled 2×2 log-mean 17/13, Pfa 0.01, gate 0.75, peak 5×5 | 0.991 | 96 |
| pooled 4×4 intensity-mean 17/13, Pfa 0.03, gate 0.6, peak 3×3 (config B) | 0.987 | 50 |

What explains it (from the phase-1 attribution): halving the resolution halves the ship's size relative to the same ring (self-masking, 9.4 points), the 2×2/4×4 averaging suppresses speckle (far fewer false events per unit of Pd), padding recovers the border ships, and a contrast-peak event keeps landing on the ship even at Pfa 3 % where the NMS corner collapses. Not helpful here: iterative censoring, 4×4 with large windows, 4×4 log-mean vs intensity-mean (equal).

**End-to-end cascade (float CNN retrained on each candidate set, DEEP architecture; thresholds chosen on validation images, applied to test):**

| | prescreen delivers an event for | cascade Pd | false events / image |
|---|---|---|---|
| old cascade (box-hit metric, INT8, measured on the FPGA) | 0.90 | 0.79–0.80 | 1.7–2.0 |
| **config A** (190 events/img) @ retention 95 / 97 / 98 / 99 % | 99.7 % | **93.5 / 95.2 / 96.4 / 98.3 %** | 2.3 / **3.5** / 5.0 / 8.8 |
| config B (50 events/img) @ 95 / 97 / 98 / 99 % | 98.5 % | 93.6 / 95.3 / 96.2 / 97.5 % | 3.1 / 4.8 / 6.1 / 9.5 |

**So the target (cascade Pd ≥ 95 %) is reached in software at ≈ 3.5 false events per image (config A) or ≈ 4.8 (config B, 4× fewer candidates for the CNN).** Cascade Pd = ships with an accepted on-ship event / all ships; FA = accepted events on no ship.

**Caveats before this is believed outside the lab:**
1. Float CNN, float prescreen (log-mean pooling of float log amplitude, float Weibull). Not yet INT8 / bit-exact; the hardware pooled store holds the mean of QROM codes (8-bit) and the Weibull LUTs / fixed-point arithmetic have to be re-derived for the pooled domain (phase 6).
2. Pd means "an event within 4 px of the polygon" — a CNN patch centred there sees the ship, but it is not a localisation/IoU measure, and touching ships share events.
3. Selection was on 600 train/val images and confirmed on test (tuning→test Pd differed by ≤ 0.8 points for the shortlisted configs), but only one CNN seed per config and no confidence intervals yet.
4. The old numbers come from different metrics (box hit, INT8, FPGA); the like-for-like old figure under this metric is prescreen 0.84–0.87 × retention ≈ 0.75.
5. Ships < 25 px mask area are few (64 of 16,951) and are the ones 4×4 pooling can lose; config B should be checked per size class.

**Next:** INT8 PTQ/QAT of the config-A/B CNNs and the same table with hardware arithmetic; seeds + bootstrap CIs; per-size / inshore-offshore breakdown of the cascade; hard-negative mining to push FA down; then phase 6 (pooled-domain Weibull in RTL: 400×400 line buffers, pooled log LUT).

(2026-10-03) A scientific-style write-up of phases 1-4 (loss attribution, design search, whole-data-set Weibull-only result) is in `PAPER2_Weibull_prescreen_Pd_report.docx`; build script `_comparison/pd_study/build_report_docx.js`, figures `_comparison/Results/pd_study/report/`.

## 12. CNN improvements (steps 1–3 of section 11 follow-up), 2026-10-03 — config A candidates, float, test split (842 images, 2,877 ships)

Code: `_comparison/pd_study/extract_cnn_patches_pooldet.m` ('Ctx': 4×4-pooled 32×32 context patch = 128×128 px field of view + 9 side features), `cnn/train_ctx.py`, `cnn/fuse_events.py`, `cnn/fuse_simple.py`, `cnn/eval_pooldet.py`, `cnn/cnn_loss_analysis.py`. One training at a time (user's RAM limit). Thresholds from validation images, applied to test; cascade Pd = ships with an accepted on-ship event / all ships.

| model | cascade Pd @ false events/image: ~95.3 % | ~96.5 % | ~98 % |
|---|---|---|---|
| DEEP, train_hw recipe (earlier, config A) | 95.2 @ 3.45 | 96.4 @ 4.97 | 98.3 @ 8.84 |
| control: same new trainer, no context, no MIL/mining | 95.2 @ 5.17 | 96.7 @ 8.00 | 98.3 @ 13.1 |
| step 1: + context tower + side features | 95.8 @ 4.85 (98 % target) | 93.1 @ 3.03 (96 % target) | 98.1 @ 8.80 |
| **steps 1 + 3: + ship-level (top-2 events/ship) loss + hard-negative mining** | **95.3 @ 2.00** | **96.6 @ 2.98** | **97.9 @ 5.09** |

* Step 1 alone is a modest gain (≈ 6–10 % fewer false events at equal Pd). Steps 1+3 together cut false events by 42 % against the earlier DEEP and 61 % against the control at 95.3 % cascade Pd. The separate contribution of step 3 without the context tower was not run.
* **Step 2 (ship-level fusion across neighbouring events) did not help.** A gradient-boosting second stage trained on validation events was clearly worse (5.7 vs 2.0 false events at 97 % target: it learns that neighbours of ship events are positive, which also promotes false events next to ships). A simple validation-tuned neighbour-evidence term chose α = 0 (no gain): once the CNN is trained at ship level its best event per ship is already high and the false events are not systematically isolated.
* Remaining loss at the 97 % target (135 of 2,877 ships; 8 never delivered): **inshore 0.917 vs offshore 0.991**, ships < 50 px² (0.18 and 0.74), contrast dx < 1.2 nat (0.52 and 0.84), single-event ships (0.867). The context tower and ship-level training reduced false events but did **not** raise retention at a given threshold, i.e. these ships look like background to the network; 98–99 % cascade Pd therefore costs 3–5+ false events/image today.
* Caveats: float network, one seed, one test split, no confidence intervals; the context tower adds a 4×4-pooled store and a second conv tower (hardware cost not yet assessed; INT8/QAT not done). Next candidates: INT8 of the winning network, a land/inshore-specific input (e.g. a land-mask or heterogeneity channel), larger FOV for small ships, and seeds + bootstrap CIs.

## 13. Whole-data-set sweep at three false-alarm settings (2026-10-03)
`_comparison/pd_study/pd_sweep3.m` (all 5,604 images, both designs, Pfa 1e-2/1e-3/1e-4, gate 0.75, 5×5 peak events, 4 px tolerance) + `sweep3_report.py` / `sweep3_cascade.py`; reports `PAPER2_Three_Pfa_report.docx` (Weibull-only + after-CNN) and `PAPER2_Explainer_contrast_peak_and_pooling.docx` (plain words, mathematics, exact code). Outputs `Results/pd_sweep3/report/` incl. per-image tables (`per_image_weibull_only.csv`, `per_image_weibull_only_and_cascade.csv`, `per_image_pfa_decade_table.csv`).
Weibull-only strict Pd, pooled 2×2 log-mean / full-res: 1e-2 0.9847 / 0.9664; 1e-3 0.9712 / 0.8714; 1e-4 0.9451 / 0.7480; events per image 91/82/72 vs 370/286/203. Achieved pixel Pfa ≈ 4e-2 / 1.7e-2 / 0.9e-2 (above nominal, saturating).
After a per-Pfa context+MIL CNN (test split, 97 % val-retention target): Pd 0.943 @ 2.07 false events/img (1e-2), 0.936 @ 2.12 (1e-3), 0.901 @ 1.48 (1e-4); at ~3 FA/img 0.956 / 0.948 / 0.915. Config A (25/17, Pfa 3e-2, gate 0.6) remains better (0.953 @ 2.0 with 99.7 % delivered).
