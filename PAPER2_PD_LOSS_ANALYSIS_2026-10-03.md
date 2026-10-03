# Why the Weibull prescreen loses ships — lost Pd points and how they were found (2026-10-03)

Scope: Paper 2 cascade prescreen = bit-exact hardware model of the Weibull CFAR (SLI 17, guard 13, Pfa plane 1e-3) + NMS trigger + gate `x − c1 ≥ 0.75`,
on **all 5,604 HRSID images / 16,951 ships**. Code: `_comparison/pd_study/` (`pd_study_extract.m`, `analyze_misses.py`); tables/figures: `_comparison/Results/pd_study/`.
Roadmap this feeds: `PAPER2_PD_ROADMAP_weibull-prescreen-to-near-100.md`.

## 1. Result

**Prescreen Pd that matters for the cascade = 0.843** (2,664 of 16,951 ships never deliver an event on the ship to the CNN). The commonly quoted "~90 %" (box hit, hardware 0.891) is a lenient number — see §2.
Offshore 0.861, inshore 0.823; train 0.838 / val 0.849 / test 0.860.

| # | Cause of the loss (mutually exclusive, priority A→E) | ships | **Pd points lost** | share of losses |
|---|---|---|---|---|
| D1 | **Own ship inside the CFAR reference ring (self-masking)** — would be detected against clean sea | 1,586 | **9.36** | 59.5 % |
| C | Gate / NMS — ship pixels are detected but no gated trigger lands on the ship | 323 | 1.91 | 12.1 % |
| D2 | Neighbouring ship inside the ring (masking) | 257 | 1.52 | 9.7 % |
| B | RTL border — float model finds the ship, RTL never evaluates the outer 8 px | 233 | 1.37 | 8.8 % |
| E1 | Low contrast, near miss (clean-sea margin −0.15…0 nat) | 135 | 0.80 | 5.1 % |
| E2 | Low contrast, deep (below the clean-sea threshold by > 0.15 nat) | 69 | 0.41 | 2.6 % |
| D3 | Bright clutter inside the ring | 53 | 0.31 | 2.0 % |
| D4 | Other (variance / shape mismatch) | 8 | 0.05 | 0.3 % |
| | **Total** | **2,664** | **15.72** | 100 % |

Key facts behind the table
* For **91 % of the lost ships the clean-sea margin is positive**: if c1 and c2 were estimated from ship-free sea, the ship would be detected. The loss is mostly an estimator/geometry problem, not invisibility. Truly low-contrast ships (E1+E2) are 1.2 points; D3+D4 0.4.
* Self-masking (D1): every such ship is wider than the 13 px guard (median longest box side 46 px); median fraction of the 120 reference-ring cells that lie on the ship at its best pixel = 0.27. Pd does not vary with size class (≤13 px 0.82, 14–26 px 0.845, >26 px 0.843) — the loss is "a bright ship that fills the ring", not "large ships only". D1 is more common offshore in absolute numbers (924 vs 662 ships); D2, E1, E2 and D3 are overwhelmingly inshore (248/9, 132/3, 69/0, 40/13).
* **Border** costs more than category B shows: within 8 px of the image edge 788 of 2,491 ships are lost (Pd 0.68 vs 0.843 overall); 29.6 % of all losses are border ships (14.7 % of ships are). Category B is only the part the float model would recover by padding; the rest of the border losses are spread over D1/C/D2/E.
* Recoverable by relaxing the threshold only (float model, gated event on the ship, among all ships): Pd 0.858 @1e-3 → 0.905 @3e-3 → **0.940 @1e-2** → 0.933 @3e-2 → 0.838 @1e-1 (the NMS corner event stops landing on the ship at high Pfa, so the event definition becomes the limit). Gated events per image stay ≈ 270–290 up to 1e-2 because the gate caps the workload.

## 2. How it was found (method)

**2.1 Measure every ship, not just the average** — `pd_study_extract.m` runs, for each image, the float Weibull model (`cfar_front_end` + `WeibullCFAR_TLog`, Pfa 1e-1…1e-6, so any threshold can be tested) **and** the bit-exact fixed-point model (`cfar_front_end_fixed`, the RTL arithmetic, planes 1e-3…1e-6 with the hardware gate `x_code + 19866 − 2·c1_code ≥ 12288`). Ground truth: HRSID polygon masks (`poly2mask`, box fallback) and boxes. Quantities in log-amplitude units `x = ln√(I+0.5)` ("nat").

**2.2 Hit definitions (the first finding).** Four definitions were computed on the same ships:
* `boxHit` — any detected pixel inside the GT rectangle (what `computePdPfa`/`metrics.py` count);
* `maskHit` — a detected pixel on the ship polygon;
* `evComp` — a gated trigger in a detection component touching the box (the CNN training-label rule);
* **`evMask` — a gated NMS trigger on the ship polygon dilated by 3 px = the event the CNN actually receives for this ship.** This is the Pd that bounds the cascade and is the one used for the loss accounting.
At Pfa 0.1 box-hit Pd = 100 % although 10 % of all pixels are flagged (a box catches sea clutter); `evMask` there is 0.84. Hardware at 1e-3: box 0.891 / pixel-on-ship 0.846 / evMask 0.843.

**2.3 Clean-sea counterfactual (the second finding).** For each ship: background samples = the box expanded by 20 px minus all ship boxes dilated by 3 px; μ_bg, σ_bg, `c2Bg = var(background)`;
`δ_clean = WeibullCFAR_TLog(WeibullCFAR_Params(c2Bg), 1e-3)`; **`marginClean = x_max,ship − (μ_bg + δ_clean)`**. `marginClean > 0` means a CFAR fed with ship-free sea statistics would detect the ship's brightest pixel.
(The first version of this study reused the CFAR's own δ, computed from the contaminated c2, and wrongly reported 96 % "window inflation" — corrected here by recomputing δ from clean sea.)

**2.4 Where the contamination comes from.** At the ship's best pixel (argmax of `x − T₁e-3` over polygon ∩ valid region) the 120 reference-ring cells (17×17 minus 13×13) are classified as: on the ship's own polygon / on another ship's polygon / non-ship pixels brighter than μ_bg + 2σ_bg ("bright clutter"). The dominant class with ≥ 5 % of the ring names the contamination; < 5 % → "other".

**2.5 Exclusive taxonomy** (`classify()` in `analyze_misses.py`; later rules override earlier ones, so the priority is A > B > C > D > E): a ship is "lost" when `evMaskFx1 = 0` (hardware model, plane 1e-3).
* **B** float `maskHit` = 1 but hardware `maskHit` = 0. (Checked: all 233 lie within 8 px of the edge, 231 within 4 px: the float front end evaluates the border, the RTL's valid region excludes the outer `TK = 8` px. It is **not** quantisation error — an earlier note in the roadmap that said "~1.4 points are fixed-point arithmetic" was wrong and is corrected there.)
* **C** hardware `maskHit` = 1 (ship pixels detected) but no gated trigger on the ship (gate `x − c1 < 0.75` at the trigger, or the NMS corner sits off the ship).
* **D** not detected but `marginClean > 0`; split by the ring contamination of §2.4 (D1 own / D2 neighbour / D3 bright clutter / D4 other).
* **E** not detected and `marginClean ≤ 0`; E1 if > −0.15 nat (near miss), E2 otherwise.
* A (no CFAR-valid pixel at all) occurred for 0 ships.

**2.6 Which factors predict a loss.** Pd by bins of size, contrast, local clutter, neighbour gap, border distance, ships per image, scene (HRSID inshore/offshore lists), with Wilson 95 % CIs; and a gradient-boosting model (train+val images → held-out test images, AUC 0.93, permutation importance). Highest: ship contrast dx, ring contamination (own / bright / neighbour), border distance; scene and ships-per-image add little once those are known. Strict Pd by factor: touching neighbours (<0.5 px gap) 0.66; within 4 px of the edge 0.66; contrast dx < 1.0 nat ≈ 0.4; ships < 25 px 0.28; very flat local clutter (sd < 0.3) 0.65.

## 3. Reproduce

```
# MATLAB (5604 images, ~35 min in 3 parallel processes; ranges split the image list)
matlab -batch "addpath('F:/Projects/CFAR'); cfar_setup(); cd('F:/Projects/CFAR/_comparison/pd_study'); pd_study_extract('Range',[1 1900],'OutName','pd_study/part_v2p1.mat')"
#   ... same for [1901 3800] -> part_v2p2.mat and [3801 5604] -> part_v2p3.mat
python _comparison/pd_study/analyze_misses.py          # tables: phase1_miss_taxonomy.csv, phase1_pd_by_factor.csv, phase1_ship_table.csv (one row per ship incl. reason)
```
Figures: `fig_p1_miss_taxonomy.png`, `fig_p1_pd_vs_pfa.png`, `fig_p1_pd_by_factor.png`, `fig_p1_contrast_vs_delta.png`.

## 4. Caveats

* Ship polygons from HRSID are approximate (±1 px; the +1 pixel-centre offset in `poly2mask` is assumed); `evMask` uses a 3 px tolerance. A few ships near each other share contamination, so D1/D2 is a dominant-class label, not an exact decomposition.
* The clean-sea counterfactual uses sea statistics from a 20 px ring around the box; strongly non-homogeneous scenes (ports) make it less reliable — those ships are mostly inshore and fall in D2/E.
* Float-model numbers (e.g. the Pfa-relaxation curve) are not bit-exact; only hardware-model numbers (planes ≥ 1e-6) are RTL-faithful. The Pfa sweep above 1e-3 therefore needs new LUT planes before it can run on the board.
* Lost-point shares are for plane 1e-3, gate 0.75; other operating points will shift them (re-run `analyze_misses.py` after changing `MISS`/`PFX` in its CONFIG).
