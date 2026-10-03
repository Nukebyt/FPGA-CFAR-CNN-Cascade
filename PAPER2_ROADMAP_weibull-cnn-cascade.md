# Paper 2 — Weibull-CFAR Prescreen + Lightweight CNN Discriminator, on FPGA

## UPDATE 2026-10-02 (later) — RTL built, verified and fit; hardware-faithful retrain

The CNN stage and the whole cascade front end now exist as RTL (`rtl/cnn/`, `rtl/cascade/`): INT8 `cnn_core`
(bit-exact to the Python integer model), NMS trigger + gate, pooled frame store, patch fetch, two-pass controller, DE10
board wrapper. System tests pass against an independent Python reference. Quartus at 800x800 with DEEP: 52% ALM,
77% M10K, 42% DSP, closes 50 MHz. Training data was rebuilt to the exact hardware spec (not the idealised pipeline) and
the models retrained; on bit-exact fixed-point events DEEP gets 87.9% ship retention at 1.98 FA/img (CFAR alone ~1,600),
~107x fewer candidates, ~78.5% end-to-end ship recall. **Throughput was then raised 5.4-6.5x (quad-pixel core + 100 MHz CNN clock): DEEP 2.5 -> 14.4 frames/s at 800x800, SMALL 4.2 -> 22.5, XL 1.0 -> 6.8 (measured cycles; bit-exact; Quartus closes both clocks).
**2026-10-03 (later): 250-image board sweep done and a layout bug fixed.** A JTAG-fed 800x800 cascade (`rtl/cascade/jtag/`) ran 250 held-out images x 4 Pfa planes on the DE10; hardware logits == Python integer model on all 178,769 events, hardware event lists == model on all 1000 frames; cascade (DEEP, plane 1e-3, CNN @90%): Pd 0.803 vs Weibull-only 0.904, pixel Pfa 2.0e-5 vs 4.5e-3, 1.7 vs 1612 false events/image; figures in `_comparison/Results/sweep250/fig_*_hw.png`. The sweep exposed that the deployed CNN had been fed transposed windows (training layout vs row-major `patch_fetch`); fixed exactly by transposing the exported weights (`transpose_export.py`, `genq_hw_*_T`), no retraining. Details: study §10.

**Board bring-up DONE 2026-10-03: the two-clock quad-core `.sof` (DEEP) was programmed on the DE10-Standard and the default-switch display matched the simulation (24 candidates / 7 accepted, LEDR[9:6]=0111).** Open: tau-sweep check on the board, power.** Details: `PAPER2_CNN_HW_STUDY_2026-10-02.md` §9, `rtl/cascade/README.md`.

## UPDATE 2026-10-02 — CORRECTION: earlier CNN/reduction numbers were measured on the wrong data. Full write-up: `PAPER2_CNN_HW_STUDY_2026-10-02.md`

Three things in the 2026-09-25/28 text below were wrong; **do not cite the v1/v2/v3 numbers
(68.3% precision, 6.5× reduction, 12.6% candidate purity, 187.8 candidates/image) anywhere**:

1. **Prevalence.** `extract_cnn_patches.m` capped negatives at 8× positives per image. Weibull at
   Pfa=1e-3 really emits **~1,700 clusters/image, only 1.2–2.0% ship-overlapping** (full
   1,200-image extraction: 2,002,303 clusters, 23,533 positive). The 12.6% / 187.8 figures were
   properties of the subsample.
2. **Geometry.** The dataset used `sli=61/guard=49`; the DE10 RTL is `SLI=17/GUARD=13`. The CNN is
   now trained on the deployed prescreen's candidate stream.
3. **Patch source.** The 17×17 Weibull window cannot supply a 32/64-px patch; the CNN needs its own
   8-bit row buffer (64 rows × W). Now in the budget.

**Replaced by** (all on HRSID, image-disjoint splits, hardware geometry, true prevalence, ship-level
metric — FA/image at fixed ship retention):
- Free gate (centre-3×3 mean − `c1` ≥ 0.45) passes 13% of candidates (229/img) → ~7.5× less CNN work.
- Biggest model gains were **data/label/context**, not size: ship-level labels (train on the 2 best
  fragments per ship) ≈ 2×; 64-px context (2×2-pooled to a 32×32 net input) ≈ 20% *with fewer MACs*;
  offset-jitter is mandatory (2.5× degradation at 3 px shift without it); 3×3-stack depth ≈ 15–35%.
  4× width: nothing. The 16k-param sizing was never the real limit — M10K/DSP have large headroom;
  *throughput at the true candidate rate* is the binding constraint.
- **Full HRSID** (5,604 images; test = 842 images / 2,877 ships). Three exported INT8 models
  (integer-exact, golden vectors, hex weights in `Results/cnn_weights_hw/full_{small,deep,xl}/`):
  SMALL 16.3k params / 0.49M MAC, DEEP 65.6k / 1.95M, XL 147k / 4.15M. Val-selected threshold → test:
  **XL 89.6% ship retention (CI 87.4–91.5) at 1.51 FA/img (1.25–1.75); DEEP 89.4% at 1.81; SMALL 87.1%
  at 2.68** — against **1,653 FA/img for CFAR alone** and ~40 FA/img for the free gate alone. ~130×
  fewer candidates (1,674 → ~12.5/img); end-to-end ship recall ≈ 81% of all GT ships (CFAR-reachable
  ceiling 90.6%). PTQ→INT8 costs ≤ ~15% in FA; QAT ≈ PTQ. Throughput (model, 128 lanes): SMALL ≈ 38
  img/s, DEEP ≈ 10, XL ≈ 4.7 (800×800 frames) — an accuracy/throughput Pareto, not one model.
- Constraint recheck, measured by Quartus: Weibull @ SLI=17, **W=800** = 17,411 ALM (42%), **118 M10K
  (21%)**, 13 DSP. Everything CNN-side (ALM/M10K/DSP/throughput) is a *model* (`cnn/hwcost.py`), not a fit.
- Not yet done / still open: RTL MAC array + trigger + patch buffer + gate, co-resident Quartus fit
  (only the Weibull row is Quartus-measured: W=800 → 118 M10K, 42% ALM, timing closes at 50 MHz),
  first-stage reduction vs a no-CFAR sliding window, prescreen Pfa/SLI sweep for the cascade, SSDD
  generalisation, CA/OS-CFAR prescreen comparison on this metric, models > 4.15M MAC / FOV > 64 px /
  distillation (untried; the 14.5M-MAC ceiling-teacher run was aborted and is not evidence).
- **Model decision (was open): SMALL (throughput point), DEEP (default), XL (accuracy point) are all
  exported; pick per the frame-rate the paper wants to claim. The 16k-param v1 is no longer the
  target.** v2/v3 (hand-feature fusion) are dropped: per-patch statistics and the 8-feature head are
  not needed and cost RTL. Capacity helps once the data is big enough (subset looked saturated; full
  HRSID is not).

## UPDATE 2026-09-25 — a directly relevant prior-art collision found, and a resulting pivot

While scoping the CNN discriminator's implementation route (RTL accelerator
vs. vendor IP), a vault search surfaced Mahoor, "FPGA Acceleration of
Automated Ship Detection and CNN-based Ship/Iceberg Discriminator in SAR
Imagery" (SFU MASc thesis, April 2023, advisor Jie Liang; full text already
in the research vault, note `etd22415pdf`). **This is the single most
directly on-point prior-art source found for this paper's core claim** —
read it before drafting anything further:

- It already puts BOTH a CFAR-analog detection stage (a novel Trimodal
  Discrete/3MD sea-clutter model, not Weibull) AND a CNN discriminator on
  FPGA — but on **two different chips**: the 3MD detector on a large
  data-center-class Xilinx Virtex UltraScale+ XCVU13P, and the CNN
  discriminator (binary ship-vs-iceberg) on a **Xilinx Zynq UltraScale+
  ZCU104 — this project's own exact original target board** — via
  Xilinx's DPUCZDX8G IP core (the vendor-IP route), driven by Vitis AI
  quantization-aware INT8 training. No accuracy loss from 8-bit
  quantization, matching what this project already assumed as a design
  premise.
- **This directly preempts a same-chip ZCU104-first version of this
  paper's plan**, specifically the "vendor IP / Vitis AI DPU on ZCU104"
  route — that combination is no longer a clean claim on that board.
- What is NOT covered by Mahoor's thesis, and remains open: (1) the two
  stages are never integrated/co-resident on ONE chip with a single
  combined resource/power/throughput budget — Mahoor's own thesis does not
  report this; (2) Weibull specifically (Mahoor uses 3MD, a heavier
  5-parameter Nelder-Mead-optimized model); (3) a true-ship-vs-false-alarm
  binary (Mahoor's CNN discriminates real-ship-vs-iceberg, i.e. both
  classes are real detected targets — a different problem from rejecting
  CFAR's false alarms); (4) any candidate-area-reduction factor (confirmed
  absent from Mahoor's thesis by direct search of its Results/Resource
  Utilization chapters — this project's own headline number is still
  unclaimed); (5) it is an unpublished MASc thesis, not peer-reviewed.

**Resulting decision, per the user's explicit instruction (2026-09-25):
build the CFAR+CNN cascade end-to-end on Intel/Altera DE10-Standard
(Cyclone V) FIRST, not ZCU104.** This is not just a workaround for the
still-broken Vivado BRAM-inference bug (item 1 below) — it is a *stronger*
differentiation than a ZCU104-first plan would have been:
- Cyclone V has no DPU-equivalent hardened AI/NPU block and no Vitis-AI-style
  toolchain, so a DE10 CNN stage is necessarily a genuine from-scratch RTL
  accelerator, not vendor-IP integration — a strictly stronger "we built
  the hardware" claim than Mahoor's own DPU-based route.
- Both stages co-resident on ONE chip, with one combined resource/power
  budget — the specific gap Mahoor's thesis leaves open.
- An entirely different vendor family (Intel/Altera vs. Xilinx) from every
  paper read so far (Mahoor, REF2, REF3), so there is no same-vendor,
  same-board overlap risk at all for the DE10 phase.
- ZCU104 remains the second phase once DE10 is working end-to-end — at
  that point the paper can honestly compare a from-scratch DE10 accelerator
  against a ZCU104 port (custom RTL or DPU-integration, decided later),
  which is a richer result than either alone.

---

**Thesis:** on-board CFAR+CNN cascades for SAR ship detection exist (REF9: K-distribution
CFAR + YOLOv4-tiny, validated on the real HISEA-1 satellite) — but every one found runs the
cascade on a GPU/embedded-SoC (Jetson TX1), never on an FPGA. This paper builds the same
class of cascade on FPGA using Weibull as the prescreen distribution, and reports the one
number no prior cascade paper reports: the candidate-area reduction factor CFAR buys the
CNN stage, in silicon.

**Target venue:** JSTARS or Sensors, per the original assessment (D9: 4.5/10 novelty,
35–45% odds at Sensors, ~80% at Access). Given the confirmed FPGA-daylight (see below),
lean toward JSTARS first — the on-board framing and real hardware numbers are closer to
what that venue already accepted for REF9's software version.

**Faculty-advised.** No authorship-politics issue here.

---

## Why "instead of K I use Weibull, and implement it on FPGA" is a real gap, not a
## small tweak

Confirmed by reading three papers in full, not by assumption:
- **REF9** (the paper you're modeling this on): Jetson TX1, ARM CPU (CFAR + sea-land
  segmentation) + 256-core Maxwell GPU via CUDA (YOLOv4-tiny). Zero FPGA anywhere in the
  paper, including its own future-work paragraph.
- **REF2** (Xu et al. 2022 JSTARS — the one real FPGA ship-detection paper with flight
  heritage, on GF-3B): zero CFAR content at all. Its detection pipeline is Otsu
  threshold + top-hat/gradient + GLCM texture + SIFT+SVM — a classical CV pipeline, not
  a statistical CFAR + CNN cascade.
- **REF3** (Gaofen-3 CFAR+CNN cascade, Rayleigh/Gamma/K prescreen + CNN discriminator):
  software only, GPU-trained (Caffe, NVIDIA K40), zero hardware.

So the specific combination — parametric-clutter CFAR prescreen, lightweight CNN
discriminator, both actually synthesized to FPGA fabric — has clean daylight. This raises
D9's novelty somewhat above the original 4.5/10 estimate, but the paper's persuasiveness
still depends on hitting the efficiency story, not the accuracy story (below).

## What already exists

- Weibull is already through full RTL, verified bit-exact, resource-probed, and
  timing-closed on Cyclone V (17,373/41,910 ALMs, 41%; 22,507 registers; 13/112 DSP;
  Fmax 55+ MHz margin at the 50 MHz target). This is your cheapest model — genuinely the
  right choice for "minimize prescreen hardware cost, maximize what's left for the CNN
  stage."
- A Vivado/ZCU104 port of Weibull was **attempted and failed at placement**
  (`_vivado/weibull/build_weibull.log`, dated 2026-09-13): synthesis reports 177,112 LUTs
  used (76.87% of the xczu7ev's 230,400) against only 176,932 registers and 3,882
  LUT-as-Memory — that ratio (huge LUT count, tiny inferred-memory count) is the exact
  signature of a BRAM-inference failure, the same class of bug the Cyclone V build hit
  and fixed via `rtl/common/row_delay_mem.v` (Quartus needed a synchronous read to infer
  M10K correctly; Vivado's BRAM inference has different, stricter coding-style
  requirements and evidently isn't recognizing the same construct). This is not a "start
  from scratch" situation — it's a known bug class with a known fix pattern on the other
  vendor, needing a Vivado-specific re-derivation.
- The HRSID/SSDD sweep already found Weibull's real (uncompressed, HRSID) peak
  performance: Pd 0.979 at `sli=61` — the highest peak Pd of any of the five original
  detectors on lossless data (finding F12, `ROADMAP.md` §6.2a). This is a genuinely
  strong number to lead the accuracy half of this paper's results with.

## What this paper still needs

### 1. Fix the Vivado BRAM-inference failure before anything else (days, not weeks — it's
### a known bug class, just on a new vendor)

- [ ] Read `weibull_top_new.xdc` and the RTL modules instantiated in
      `build_weibull.tcl` against `_vivado/weibull/output_files/post_synth.dcp` to find
      exactly which memory (`line_buffer.v`, `row_delay_mem.v`, or the moment-accumulator
      RAM) is failing to map to a Xilinx BRAM primitive and is instead getting inferred
      as LUTRAM/logic.
- [ ] Xilinx's synchronous-read requirement is similar in spirit to Quartus's but the
      exact coding idiom differs (Vivado wants a registered read address AND typically
      prefers explicit `(* ram_style = "block" *)` attributes or XPM macros over
      inference-only code for reliability). Re-derive the memory-inference idiom for
      Vivado specifically — do not assume the Quartus fix ports unchanged, the existing
      `ROADMAP.md` already flags this exact risk as unconfirmed.
- [ ] Re-run `build_weibull.tcl` after the fix and confirm utilization drops to a
      sane fraction of the xczu7ev's 230,400 LUTs (Weibull's Cyclone V share was ~41% of
      a much smaller 41,910-ALM device — on a device this much larger it should land
      well under 10% once memory infers correctly).
- [ ] Get through `place_design` and `route_design`, not just synthesis — the failure so
      far is at placement, meaning routing/timing closure numbers don't exist yet at all.

### 2. Build the CNN discriminator stage — the part that doesn't exist yet

Nothing in the repo currently touches a CNN on FPGA. This is the paper's second hardware
component and needs to be scoped realistically.

- [ ] Pick a target network scale deliberately, not by default. REF9's own comparison
      table (YOLOv3/v4/v3-tiny/v4-tiny/SSD/Faster R-CNN) found YOLOv4-tiny the best
      accuracy/speed tradeoff for single-class (ship) detection at 22.4 MB — but that's
      sized for a GPU, not FPGA fabric. An FPGA CNN stage this small a device can host
      will likely need something smaller still (a compact classifier operating only on
      the CFAR-flagged candidate patches, not a full detector run over the whole image —
      this is the architectural point of a cascade: the CNN only ever sees what CFAR
      already flagged).
- [ ] Decide the FPGA CNN implementation route explicitly: a from-scratch RTL
      accelerator (most control, most effort, strongest novelty claim), or a vendor IP
      core (Xilinx DPU / Vitis AI) instantiated alongside your CFAR RTL on the same
      device (faster to build, weaker "we built the hardware" claim, but still a real
      FPGA result — REF2's HE-BiDet/ARMOR-class precedents from the earlier corpus review
      used custom accelerators, so a vendor-IP route should be framed honestly as
      integration work, not an accelerator-design contribution).
- [ ] Whichever route: get a resource-utilization number for the CNN stage ALONE, so the
      paper can report CFAR-stage cost, CNN-stage cost, and total, separately — this
      breakdown is what makes the "prescreen saves you X% of the CNN workload" argument
      legible.

**2026-09-25, decided (DE10 phase):** from-scratch RTL, forced by the platform (Cyclone V
has no DPU-equivalent). Network architecture, chosen to fit comfortably in the ~58% of
41,910 ALMs and ~88% of 112 DSP blocks Weibull leaves free (`5CSXFC6D6F31C6`):

- 32x32x1 input patch (single-channel log-amplitude, `fe.x` — the exact representation
  Weibull's own threshold decision already uses, no new front-end needed).
- Conv1 (8x 5x5) -> ReLU -> 2x2 maxpool -> Conv2 (16x 5x5) -> ReLU -> 2x2 maxpool ->
  FC1 (400->32) -> ReLU -> FC2 (32->1, sigmoid). ~16.3k params, ~490k MACs/patch.
- INT8 post-training quantization (per-tensor symmetric), matching Mahoor's own
  precedent that 8-bit costs no accuracy — reproduced independently, not assumed
  (`_comparison/Results/cnn_weights/manifest.json` records the actual fp-vs-int8 test
  delta once training runs).
- **A real gap this decision surfaces, not yet resolved:** none of the existing Weibull
  RTL does connected-component clustering — it emits a raw per-pixel detection bitmap,
  not discrete candidate patches. Real-time CCL is itself nontrivial hardware. Default
  plan: a simplified non-max-suppression trigger (first detected pixel in a bounded
  neighborhood fires a fixed-size patch fetch from the existing line-buffer window,
  with a cooldown region to avoid re-triggering on the same cluster) rather than full
  two-pass CCL — cheaper, and sufficient because what the CNN sees is discrimination-only
  (real ship vs. false alarm), not localization, so an approximate trigger centroid is
  tolerable in a way it would not be for a bounding-box-accuracy claim. Revisit if
  validation shows this misses too many ship-containing clusters.

**2026-09-28, v2/v3 iteration (still software-only, no RTL changed):** v2 (wider net +
BatchNorm + augmentation + precision-targeted model selection) trained; barely moved the
default-threshold operating point vs. v1, but delivers a full precision/recall sweep —
90.4% precision at 63.8% recall (thresh=0.90), or 83.9%/75.6% at best-F1 (thresh=0.80).
Full table in `PAPER2_DRAFT_weibull-cnn-cascade.md` §3.3. v3 (hard-negative mining via
v2's own confident-false-positive scores + an 8-feature CFAR-statistics fusion head)
launched same day, training in progress — no results yet, do not write any v3 numbers
into the draft until `Results/cnn_discriminator_v3.pt` actually exists.

### 3. Measure the number no one else reports: candidate-area reduction factor

This is the single figure the original assessment identified as absent from every
existing cascade paper (the thesis you're writing (REF9), the IGARSS paper, the thesis
citing "combining CFAR and ML methods" as future work) — and it's the number that makes
this paper's contribution measurable rather than just "we ported X to FPGA."

- [ ] **SUPERSEDED 2026-10-02 — numbers below were computed on a negative-capped, sli=61 subsample
      and are wrong for the deployed cascade; see the update at the top and
      `PAPER2_CNN_HW_STUDY_2026-10-02.md` §5 for the corrected second-stage reduction (CFAR 1,728
      candidates/img → 11.1/img at 85% ship retention, DEEP INT8).** Original text (do not cite):
      (second-stage reduction — CFAR-flagged clusters vs. CNN-accepted
      clusters). 225,376 Weibull-flagged clusters over 1,200 HRSID images (`sli=61,
      Pfa=1e-3`) = 187.8 candidates/image; only 12.6% are real ships. The trained CNN
      discriminator (§3.3) accepts 5,485 of the 35,739 held-out-test candidates — a
      **6.5x reduction** in what needs further review, retaining 87.1% of real ships and
      raising candidate precision from 12.6% to 68.3% (a 5.4x purity improvement).
      `Results/cnn_discriminator.pt`'s `test_metrics` field carries the exact counts.
- [ ] **Still open (first-stage reduction — CFAR prescreen vs. a hypothetical full
      sliding-window CNN pass over the whole image, no CFAR at all).** This is the number
      directly comparable to REF9's own reported 673 candidate patches on their large test
      image — needs HRSID's typical image dimensions and a defined sliding-window
      stride/size convention for the "no CFAR" baseline before it can be computed; not
      yet done. Do not conflate this with the second-stage number above when writing §5 —
      they answer different questions and this paper should report both.
- [ ] Final detection accuracy (Pd/F1) for the full cascade, using the fixed metric from
      Paper 1's item 3 — not yet computed end-to-end (the 93.58%/68.3%/87.1%
      accuracy/precision/recall numbers above are the CNN stage's OWN classification
      metrics on Weibull-flagged patches, not a full-image Pd/F1 cascade evaluation).

### 4. Run CA-CFAR / OS-CFAR baselines before any detection-accuracy claim (BC-5, shared
### with all hardware papers)

No CA-CFAR or OS-CFAR implementation exists anywhere in this repo (confirmed — no match
for either in `_comparison/` or `_common/`). This is the textbook comparator every
reviewer will ask for, and it's needed here specifically because the paper's cascade
claim ("CFAR narrows the field, CNN cleans up") needs a baseline showing what a
non-parametric CA/OS-CFAR prescreen would have cost in false alarms fed to the CNN.

- [ ] Implement CA-CFAR (and ideally OS-CFAR) in the existing MATLAB comparison harness
      (`_comparison/run_comparison_hrsid.m` pattern), reusing the shared front end where
      possible — this is ~1 week of work per the original effort estimate, and it directly
      strengthens this paper's cascade-efficiency argument (a naive CA-CFAR prescreen
      would pass many more false candidates to the CNN than Weibull does).

## What the paper must NOT do

- Do not frame this as "we maximized Weibull's detection accuracy" as the headline —
  REF9 already gets 88.6% recall with K-CFAR + YOLOv4-tiny on real satellite data; an
  accuracy-only story invites a direct, unflattering comparison. Lead with the
  efficiency/reduction-factor number and the fact that it's on FPGA, not GPU.
- If citing REF9's or the Kria/Jetson power figures for context (BC-7), report your own
  device's power draw too, even as a vendor-tool estimate — never cite a competitor's
  wattage while leaving your own table silent on it.

## Submission-readiness gate

- [ ] Vivado Weibull port through placement and routing with real utilization/timing
      numbers (item 1)
- [ ] CNN discriminator stage built and resource-probed, with an explicit statement of
      which implementation route was taken (item 2)
- [ ] Candidate-area reduction factor measured and reported (item 3)
- [ ] CA-CFAR/OS-CFAR baseline run and included (item 4)
- [ ] Detection-accuracy numbers sourced from the fixed `cfar_metrics.m` (see Paper 1
      roadmap item 3) — do not let this paper ship a number from the pre-fix metric
- [ ] Own-device power figure reported if any competitor's wattage is cited
