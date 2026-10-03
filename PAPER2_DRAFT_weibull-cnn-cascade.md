# Weibull-CFAR Prescreen + Lightweight CNN Discriminator, on FPGA

**Status tracker.** Written incrementally as roadmap items in
`PAPER2_ROADMAP_weibull-cnn-cascade.md` are completed. `[PLACEHOLDER -- ...]`
marks a section where the underlying measurement doesn't exist yet — never
filled with invented numbers.

**Target venue:** JSTARS primary (given confirmed FPGA daylight vs. REF9/
REF2/REF3), Sensors fallback. Faculty-advised.

> **2026-10-02 CORRECTION — read before using anything in §1, §3.3 or §5.** The v1/v2/v3 CNN
> numbers below (12.6% candidate purity, 187.8 candidates/image, 68.3% precision, **6.5× reduction**,
> F1/precision sweeps) were measured on a dataset whose negatives were capped at 8× positives and
> whose prescreen window (`sli=61`) is not the one in hardware (`SLI=17`). The real Weibull candidate
> stream is ~1,700 clusters/image at 1.2–2.0% prevalence. They are **superseded** by the ship-level,
> true-prevalence, hardware-geometry study in `PAPER2_CNN_HW_STUDY_2026-10-02.md` (DEEP/SMALL INT8
> models; e.g. full-HRSID test, DEEP INT8: 1,674 → 12.7 candidates/image at 89.4% ship retention,
> 1.81 false alarms/image vs 1,653 for CFAR alone). Old text is kept below only as history; the §3.3
> rewrite and the §5 results are still to be done from the corrected numbers.

**2026-09-25 pivot:** build the CFAR+CNN cascade on Intel/Altera DE10-Standard
(Cyclone V) FIRST, then ZCU104 — see
`PAPER2_ROADMAP_weibull-cnn-cascade.md`'s 2026-09-25 update for the full
reasoning (a directly relevant prior-art collision, Mahoor's SFU MASc
thesis, preempts a same-chip ZCU104-only plan; DE10-first is a stronger,
not merely a fallback, differentiation).

---

## Roadmap item tracker

| Item | Status |
|---|---|
| Vivado BRAM-inference fix (item 1) | [ ] Not started — deferred behind the DE10 phase; needs a Vivado build, will ask before running |
| CNN discriminator stage (item 2) | [x] v1 trained + INT8-quantized + RTL weights exported (2026-09-25); [x] v2 (wider/BatchNorm/augmentation, precision-focused) trained 2026-09-28; [x] v3 (hard-negative mining + CFAR-statistics feature fusion) trained 2026-09-28/29 -- genuine Pareto improvement over v2 (F1 0.813 vs. 0.772 at default threshold; see §3.3). RTL accelerator, the patch-trigger/clustering logic, and INT8 quantization for v2/v3 still NOT built |
| Candidate-area reduction factor (item 3) | [x] Second-stage (CFAR-flagged -> CNN-accepted) done: v1 6.5x/87.1% retention/12.6%->68.3% precision; v3 improves this to 12.6%->78.9% precision at 83.8% retention (default threshold) with a full tunable operating-point sweep (see §3.3). First-stage (vs. a no-CFAR sliding-window baseline, comparable to REF9's 673) still open |
| CA-CFAR/OS-CFAR baseline (item 4) | [x] CA-CFAR built (shared with Paper 3's BC-5); OS-CFAR not started; not yet run on this paper's own grid |
| `cfar_metrics.m` fixed before any F1 number ships (gate) | [x] Fixed and verified (D23) |
| Own-power disclosure (gate, if citing REF9/Kria wattage) | [ ] Not started |
| GammaTex-MAP7 (two-stage Gamma-MAP despeckle + Gamma CFAR) added to the shared 10-detector comparison registry | [~] HRSID regeneration in progress (2026-09-28); SSDD regeneration not yet started. Relevant to this paper only insofar as it shares `detector_registry.m`/the sweep infrastructure with Papers 1 and 3 — GammaTex-MAP7 is not itself part of this paper's cascade |

---

## 1. Introduction

CFAR-then-CNN cascades for SAR ship detection are not new: a parametric or
semi-parametric clutter model flags candidate regions cheaply, and a CNN
discriminates real ships from clutter false alarms only within those
regions, avoiding a full-image CNN pass. REF9 validates this pattern on
real HISEA-1 satellite data (K-distribution CFAR + YOLOv4-tiny, 88.6%
recall) — but runs entirely on a Jetson TX1's ARM CPU and GPU, never on an
FPGA. Two lines of prior work put pieces of this cascade on FPGA
separately: REF2 flies a classical (non-CFAR, non-CNN) computer-vision
detection pipeline on a real FPGA aboard GF-3B; more directly, Mahoor's 2023
MASc thesis puts BOTH a statistical CFAR-analog detector and a CNN
discriminator on FPGA — but on two different chips, a large data-center-
class Virtex UltraScale+ for detection and a Zynq UltraScale+ ZCU104 for
the CNN (via Xilinx's DPU IP), with no reported single-chip integration or
combined resource budget.

This paper closes that specific, remaining gap: a Weibull-CFAR prescreen and
a CNN discriminator, co-resident on ONE chip, with one combined resource
budget — built first on Intel/Altera Cyclone V (DE10-Standard), where no
vendor DPU-equivalent exists, forcing (and demonstrating) a genuine
from-scratch RTL CNN accelerator rather than IP integration, then ported to
ZCU104. Three choices distinguish this paper's cascade from every prior one
found: **Weibull**, not a heavier multi-parameter clutter model, as the
prescreen (the cheapest of six clutter families this research program has
characterized on this exact hardware target); a **true-ship-vs-false-alarm**
discrimination task, not Mahoor's ship-vs-iceberg (both real targets); and
the **candidate-area reduction factor** the CFAR stage buys the CNN stage —
a number this paper measures directly and which, to the search this project
has conducted, no prior cascade paper reports for itself.

**This paper does not lead with detection accuracy.** REF9 already reports
88.6% recall with a heavier detector-plus-YOLOv4-tiny pipeline on real
satellite data; an accuracy-only pitch invites a direct and unflattering
comparison. The headline result instead is efficiency, measured concretely:
**[CORRECTED 2026-10-02 — full HRSID, 842-image held-out test split; hardware models are INT8 and
integer-exact in simulation, not yet in RTL]** Weibull at Pfa=1e-3 (SLI=17, the hardware window) emits
~1,670 candidate clusters per image, ~1–2% of which touch a ship. A free contrast gate plus the INT8 CNN
discriminator reduces that to ~12.7 candidates per image while retaining 89% of the ships the prescreen
reached (DEEP model, 1.81 false alarms per image vs 1,653 for CFAR alone and ~40 for the gate alone) —
a ~130x reduction in what a downstream system (or a human analyst) would review — on hardware small
and cheap enough to fly, not a data-center GPU.

## 2. Related Work

**REF9** — the closest work in spirit, and the one this paper's own cascade
structure is modeled on: K-distribution CFAR prescreen plus YOLOv4-tiny CNN
detector, on a Jetson TX1 (ARM CPU for CFAR and sea-land segmentation, a
256-core Maxwell GPU via CUDA for YOLOv4-tiny), validated on real HISEA-1
satellite imagery at 88.6% recall. Zero FPGA content anywhere in the paper,
including its own future-work section. This paper's contribution relative
to REF9 is entirely the hardware target: the same cascade *class*, on
silicon small enough to fly on-board, not a GPU-equipped host.

**REF2** (Xu et al., GF-3B) — the one real flight-heritage FPGA ship-
detection paper found in this search, but its detection pipeline is
classical computer vision (Otsu threshold, top-hat/gradient filtering, GLCM
texture, SIFT+SVM), not a statistical CFAR model and not a CNN. It
establishes that FPGA-based on-board SAR ship detection is an accepted,
flown approach, but shares no architecture with a CFAR+CNN cascade.

**REF3** (Gaofen-3 CFAR+CNN cascade) — architecturally the closest match to
this paper's cascade concept (parametric clutter-model prescreen —
Rayleigh/Gamma/K — feeding a CNN discriminator), but entirely software,
GPU-trained (Caffe, NVIDIA K40), with no hardware implementation of either
stage.

**Mahoor (2023, SFU MASc thesis, advisor Jie Liang)** — the single most
directly on-point prior work found, and read in full, not just abstracted.
Proposes a novel Trimodal Discrete (3MD) sea-clutter model with a
hardware Nelder-Mead Simplex parameter optimizer as the CFAR-analog stage,
implemented on a Xilinx Virtex UltraScale+ XCVU13P (a large, data-center-
class part): 4 parallel detection engines, 296k LUTs (17.2%), 2.4ms
latency, 414 FPS, a 27.07x throughput improvement over a CPU software
baseline, 100% recall on the tested sample window. Separately, a CNN
ship-vs-iceberg discriminator (not a false-alarm rejector — both classes
are real detected targets), INT8-quantized via Vitis AI with no measured
accuracy loss, deployed via the Xilinx DPUCZDX8G IP core on a **Zynq
UltraScale+ ZCU104 — this project's own original target board** — 46,223
LUTs (21%), 690 DSPs (40%). **What is not in this thesis, and is this
paper's remaining gap to fill:** the two stages are characterized on two
different chips with no reported combined, single-chip resource or power
budget; no candidate-area-reduction factor is reported anywhere (confirmed
by direct search of its own Results and Resource Utilization chapters); the
CNN task is ship-vs-iceberg, not the false-alarm-rejection framing a
CFAR-then-CNN cascade against clutter needs; and it is an unpublished
MASc thesis, not a peer-reviewed venue.

## 3. Architecture

### 3.1 CFAR prescreen stage

*Verified, from this project's existing Weibull hardware work:* Weibull is
already through full RTL, verified bit-exact, resource-probed and
timing-closed on Cyclone V — 17,373/41,910 ALMs (41%), 22,507 registers,
13/112 DSP, Fmax margin +3.046 ns at the 50 MHz target. This is the
cheapest of the six statistical models in this project, consistent with
"minimize prescreen cost, leave silicon budget for the CNN stage."

A real-data accuracy figure worth leading the prescreen's accuracy story
with, once confirmed still valid under the corrected metric (see §5): the
HRSID sweep found Weibull's peak real (uncompressed) Pd = 0.979 at `sli=61`
— the highest peak Pd of any of the five original detectors on lossless
data (`FINDINGS.md` F12).

### 3.2 Non-parametric baseline for the prescreen comparison

*Shared with Paper 3, see that paper's §3.2 for the full derivation.*
CA-CFAR (`_common/CACFAR_Floating.m`) is now available in the same sweep
infrastructure this project already uses, so this paper's cascade-efficiency
argument ("a naive CA-CFAR prescreen would pass many more false candidates
to the CNN than Weibull does") can be measured directly rather than argued
qualitatively, once the sweep is actually run (see §5's placeholder).

### 3.3 CNN discriminator stage

> **SUPERSEDED 2026-10-02 — historical record only; numbers in this subsection are not valid for the
> deployed cascade (capped-negative, sli=61 dataset). See `PAPER2_CNN_HW_STUDY_2026-10-02.md`.**

*Built 2026-09-25 (DE10 phase). Real-ship-vs-false-alarm binary classifier,
trained and quantized, not yet in RTL.*

**Architecture, chosen to fit the ~58% of ALMs and ~88% of DSP blocks
Weibull's own Cyclone V build (`5CSXFC6D6F31C6`) leaves free:** 32x32x1
input (single-channel log-amplitude, the exact `fe.x` representation
Weibull's own threshold decision already uses); Conv1 (8x 5x5) -> ReLU ->
2x2 maxpool -> Conv2 (16x 5x5) -> ReLU -> 2x2 maxpool -> FC1 (400->32) ->
ReLU -> FC2 (32->1, sigmoid). 16,289 parameters, ~490k MACs/patch.

**Training data.** 225,376 patches extracted from 1,200 HRSID images via the
real Weibull prescreen (`sli=61/guard=49` — Weibull's own measured HRSID
peak-Pd geometry — at a deliberately loose `Pfa=1e-3`, since a cascade's
CFAR stage does not need to be stingy when a discriminator stage follows
it), each cropped around a detected 8-connected cluster's centroid and
labeled by the identical ground-truth-box overlap test `cfar_metrics.m`
already uses. 28,317 positive (real-ship) / 197,059 negative (false-alarm)
patches — a **12.6% raw positive rate**, i.e. only 1 in ~8 of Weibull's own
loose-Pfa candidate clusters is a real ship, which is exactly the gap a
discriminator stage exists to close. Class-balanced (`pos_weight`-weighted
BCE) training, 70/15/15 train/val/test split **by source image**, never by
patch, so no clutter statistics leak across the split.

**Held-out test-set result (image-disjoint, 180 images, 35,739 patches):
93.58% accuracy, 68.30% precision, 87.06% recall, F1=0.7654.** Precision
rises from the raw 12.6% CFAR-candidate purity to 68.3% post-CNN — a
**5.4x improvement in candidate quality** — while retaining 87.1% of real
ships. Framed as a candidate-count reduction: of the 35,739 test-set
candidates Weibull flagged, the CNN accepts 5,485 — a **6.5x reduction** in
what would need further (human or downstream) review, at the cost of
missing 12.9% of the ships CFAR itself found. This is this paper's central
number (see §5 roadmap item 3) computed for the first time on a real
dataset with a real trained model, not asserted.

**INT8 quantization** (post-training, per-tensor symmetric, matching this
project's own re-verification rather than trusting a precedent):
93.58% -> 93.47% test accuracy, a 0.10-point cost — independently confirms
Mahoor's own reported "no accuracy loss from 8-bit quantization" finding
for a different network on a different task. Exported as sixteen `.hex`
ROMs (`Results/cnn_weights/*.hex`, one INT8 value per line, `$readmemh`-
ready) totaling exactly the model's 16,289 parameters, plus
`Results/cnn_weights/manifest.json` recording every layer's shape and
per-tensor scale factor for the RTL fixed-point datapath.

**Not yet done (v1):** the RTL MAC-array/accelerator itself, the patch-trigger
(non-max-suppression) logic that turns Weibull's per-pixel detection bitmap
into discrete 32x32 patches for the CNN to consume (no connected-component
labeling exists anywhere in this project's RTL — see the roadmap's item 2
note on this gap), Quartus resource/timing closure, and the SSDD-side
generalization check (this model was trained and tested on HRSID only). These
gaps carry forward unchanged to v2 and v3 below — none of the software
iterations touch the RTL/hardware side.

*Reproduce (v1):* `_comparison/extract_cnn_patches.m` -> `Results/cnn_patches.mat`
-> `_comparison/cnn/train_cnn_discriminator.py` -> `Results/cnn_discriminator.pt`
-> `_comparison/cnn/quantize_export_cnn.py` -> `Results/cnn_weights/*.hex` +
`manifest.json`. Needs `h5py`, `torch` (CPU build sufficient).

**v2: wider network, precision-targeted training (2026-09-28).** v1's
93.6% headline accuracy is a misleadingly easy number on this dataset —
rejecting every candidate already scores 84.65% accuracy given the raw 12.6%
positive rate, so accuracy alone does not demonstrate the discriminator is
doing useful work. Precision — "when the cascade says ship, how often is it
right" — is the metric that actually matters for a deployed system, so v2
targets it directly rather than F1 or accuracy. Changes from v1, on the same
`cnn_patches.mat` dataset and the same image-disjoint 70/15/15 split (same
seed, so v1/v2/v3 are directly comparable): data augmentation (flip/90-degree
rotation — label-preserving, since SAR ship orientation relative to the
sensor track is arbitrary); a wider network (Conv1 16x5x5, Conv2 32x5x5, FC1
400->64, BatchNorm after each conv, Dropout(0.3) before the output layer); 60 epochs
with cosine LR annealing; model selection on validation precision subject to
a recall >= 0.80 floor (so precision cannot be optimized by silently starving
recall); and a full post-training precision/recall operating-point sweep on
the held-out test set, reported in full rather than a single cherry-picked
threshold.

**v2 held-out test-set operating points** (same 180-image, 35,739-patch test
split as v1):

| threshold | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| 0.50 | 93.64% | 67.97% | 89.26% | 0.772 |
| 0.60 | 94.40% | 72.61% | 85.94% | 0.787 |
| 0.70 | 95.00% | 77.86% | 81.71% | 0.797 |
| 0.80 | 95.31% | 83.88% | 75.55% | **0.795** (best F1) |
| 0.85 | 95.12% | 86.67% | 70.28% | 0.776 |
| 0.90 | 94.82% | **90.35%** | 63.77% | 0.748 |
| 0.95 | 93.96% | 94.30% | 53.06% | 0.679 |
| 0.97 | 93.18% | 95.96% | 45.25% | 0.615 |
| 0.99 | 91.35% | 97.42% | 28.96% | 0.446 |

At the default threshold (0.50), v2 is essentially unchanged from v1
(67.97%/89.26% vs. v1's 68.30%/87.06%) — confirming the architecture upgrade
alone (more capacity, BatchNorm, augmentation) had already converged to
roughly the same decision boundary v1 found, i.e. the ceiling was not a
capacity problem. What v2 adds is a *reportable, tunable operating curve*:
raising the threshold to 0.90 clears 90% precision (the target this paper's
cascade is being pushed toward) at the cost of recall falling to 63.8% —
retaining fewer than two in three of the ships Weibull itself flagged. 0.80
is the best-F1 compromise point (83.9% precision, 75.6% recall). Whether 0.90
or 0.80 is the "right" number to lead with depends on the downstream cost of
a missed ship vs. a reviewed false alarm — not resolved by this measurement
alone, and left as an explicit design choice rather than picked here.

*Reproduce (v2):* `_comparison/cnn/train_cnn_discriminator_v2.py` ->
`Results/cnn_discriminator_v2.pt` (same input `cnn_patches.mat` as v1; not
yet quantized/exported to RTL weights).

**v3: hard-negative mining + CFAR-statistics feature fusion (launched
2026-09-28, training in progress — no results yet).** v2's plateau at the
default threshold indicated the limiting factor was not model capacity but
what the model is shown during training, motivating two changes rather than
a further architecture search:

1. *Hard-negative mining.* Training negatives are a random 8:1-capped
   subsample per image (`extract_cnn_patches.m`) — mostly flat sea, which the
   network already separates from ships almost perfectly and learns little
   from. v2's own checkpoint is used to score every training-set negative;
   the top 25% by predicted ship-probability (the ones v2 itself is least
   sure about — wave streaks, azimuth ghosts, coastline glare) are
   oversampled 3x via a `WeightedRandomSampler`, concentrating training
   signal on the boundary cases v2's own errors identify, not on easy
   already-solved examples.
2. *CFAR-statistics feature fusion.* v1/v2 see only the raw 32x32 pixel
   patch and must re-derive contrast/texture/shape from pixels alone on a
   small network. An 8-feature vector (cluster pixel area — the one true
   CFAR-cluster statistic retained at extraction time; peak log-amplitude;
   local background estimate; contrast; patch standard deviation; patch
   skewness; and an approximate footprint aspect ratio/area re-derived by
   thresholding the patch itself, since the true connected-component mask
   was not kept at extraction time) is fused with the CNN's 64-d image
   embedding through a small MLP head before the final classification layer.
   The footprint-shape features are explicitly flagged as approximate (a
   patch-local re-derivation, not the original cluster mask) — the
   pixel-statistic features (contrast/std/skew) are exact.

Both changes reuse v2's exact train/val/test image split for a fair
before/after comparison.

**v3 held-out test-set operating points** (same 180-image, 35,739-patch test
split as v1/v2; model: 67,377 parameters, best epoch 54/60, selected on
validation precision at recall >= 0.80):

| threshold | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| 0.50 | 95.35% | 78.91% | 83.80% | **0.813** (best F1) |
| 0.60 | 95.54% | 82.64% | 79.67% | 0.811 |
| 0.70 | 95.56% | 86.12% | 75.25% | 0.803 |
| 0.80 | 95.23% | 89.53% | 68.37% | 0.775 |
| 0.85 | 94.94% | 91.21% | 64.14% | 0.753 |
| 0.90 | 94.47% | 93.56% | 58.10% | 0.717 |
| 0.95 | 93.62% | 96.08% | 49.04% | 0.649 |
| 0.97 | 92.99% | 97.12% | 43.06% | 0.597 |
| 0.99 | 91.75% | 98.78% | 31.91% | 0.482 |

**This is a genuine improvement over v2, not a re-shuffled trade-off.** At the
same default threshold (0.50), v3 gains ~11 points of precision over v2
(78.9% vs. 68.0%) while giving back only ~5.5 points of recall (83.8% vs.
89.3%), and F1 improves from 0.772 (v2) to 0.813 (v3). v3's threshold=0.85
operating point (91.2% precision, 64.1% recall) matches or beats v2's
threshold=0.90 point (90.4%/63.8%) at a *lower* threshold — the model's
confidence calibration itself improved, not merely where on the curve one
chooses to sit. Notably, v3's best-F1 point falls at its own default
threshold (unlike v2, whose best F1 required moving away from 0.50 to 0.80),
consistent with the diagnosis that motivated this iteration: v1/v2's
ceiling was the training signal (mostly-easy random negatives, no explicit
geometric/contrast context), not model capacity, and both hard-negative
mining and feature fusion address that signal directly rather than adding
raw capacity.

*Reproduce (v3):* `_comparison/cnn/train_cnn_discriminator_v3.py` (loads
`Results/cnn_discriminator_v2.pt` for hard-negative scoring in addition to
`cnn_patches.mat`) -> `Results/cnn_discriminator_v3.pt`. Needs `scipy` in
addition to v2's `h5py`/`torch`. **Not yet done for v3:** INT8 quantization
and RTL weight export (only v1 has been through that pipeline so far); which
version (v1, v2, or v3, and which threshold) becomes the one actually
carried into hardware is an open decision, not yet made.

## 4. Hardware Implementation

`[PLACEHOLDER -- item 1 (Vivado ZCU104 BRAM-inference fix) not started. The
2026-09-13 attempt failed at placement (177,112/230,400 LUTs used, 76.87%,
against only 3,882 LUT-as-Memory -- the exact signature of a BRAM-inference
failure, same bug class as the Cyclone V D21 fix but on a different vendor
with different coding-style requirements). This needs a real Vivado build to
diagnose and fix -- will ask before running one. No utilization/timing
numbers exist past synthesis; nothing reached place-and-route.]`

## 5. Results

*Data-integrity note (shared with Paper 3):* `_common/cfar_metrics.m`'s
TP/FN counting bug (D23) was fixed and verified 2026-09. This paper's
detection-accuracy numbers (roadmap's explicit gate: "do not let this paper
ship a number from the pre-fix metric") must be generated after this fix.
None have been generated yet.

`[PLACEHOLDER -- candidate-area reduction factor (item 3) not measured;
depends on §3.3's CNN stage existing first. CA-CFAR-vs-Weibull candidate-
count comparison (item 4) not yet run despite CA-CFAR now being available
-- needs the actual sweep executed on this paper's own image set/window
size choice, which has not been decided yet either. REF9's own reported 673
candidate patches (their large test image) is the anchor number to compare
against once this paper has its own figure.]`

## 6. Discussion

`[PLACEHOLDER]`

## 7. Conclusion

`[PLACEHOLDER]`
