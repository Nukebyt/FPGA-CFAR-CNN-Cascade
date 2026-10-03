# Findings — A Unified Comparison of Five CFAR Clutter Models for SAR Ship Detection

Research notes for a paper comparing **Weibull**, **Lognormal**, **Generalized
Gamma**, **G0** and **Burr XII** CFAR detectors, with an FPGA implementation in
view.

Every number in this document was produced by code in this repository and can be
regenerated; the "Reproduce" line under each finding says how. Findings are
labelled **F*n*** so they can be cited from the RTL, the bug log and the paper
draft. Claims that are *not* yet established are collected in
[§9 Open questions](#9-open-questions) rather than being softened into the
results.

---

## 1. The central structural result

**F1 — In the log-amplitude domain, all five detectors are the same detector
with a different additive constant.**

Let `x = log(sqrt(I + 0.5))` be the log-amplitude, and let `c1, c2, c3` be the
first three sample log-cumulants over the reference cells. Then every one of the
five decision rules can be written

```
    T_log  = c1  +  δ(shape parameters, Pfa)
    detect = x_cut > T_log
```

| Detector | Shape | Derived from | `δ = T_log − c1` |
|---|---|---|---|
| Weibull | `C` | `c2` | `K(Pfa)/C`,  `K = γ + log(−log Pfa)` |
| Lognormal | `σ` | `c2` | `z(Pfa)·σ`,  `σ = √c2`,  `z = √2·erfcinv(2Pfa)` |
| Generalized Gamma | `k, v` | `r = c3²/c2³`, then `c2` | `[log Γ⁻¹(Pfa, k) − ψ(k)] / v` |
| G0 | `L, α` | `c2, c3` | `½[ψ(u) − ψ(L) + log(x/(1−x))]`,  `u = −α` |
| Burr XII | `κ, ρ` | `s = c3/c2^{3/2}`, then `c2` | `[ψ(κ) − ψ(1) + log(Pfa^{−1/κ} − 1)] / ρ` |

Two of these forms are non-obvious and are derived in this project rather than
taken from the source scripts:

**Generalized Gamma.** From `σ = exp(c1 − (ψ(k) − log k)/v)` and
`T = σ·((1/k)·G)^{1/v}`, the two `log k` terms cancel exactly:

```
    log T = c1 − (ψ(k) − log k)/v + (log G − log k)/v
          = c1 + (log G − ψ(k)) / v
```

**G0.** Substituting the F-quantile through the incomplete beta,
`Fq = (u/L)·x/(1−x)`, into
`log T = c1 + ½[−ψ(L) + ψ(u) + log L − log u + log Fq]` cancels **both**
`log L` and `log u`:

```
    δ = ½ [ ψ(u) − ψ(L) + log( x/(1−x) ) ]
```

so the G0 scale parameter `γ` never has to be formed at all.

### Why this matters

1. **`c1` carries the entire clutter level.** `δ` depends only on shape and
   `Pfa` — never on image brightness. That is exactly what makes `δ`
   precomputable into a read-only table.
2. **The expensive front end is shared.** Line buffers, window sums and the
   moment accumulators are *bit-identical* across all five detectors. Only a
   small back-end table changes. This is the difference between one accelerator
   with a swappable ROM and five separate accelerators.
3. **It makes the comparison fair.** Because the five differ *only* in `δ`,
   feeding them bit-identical `c1/c2/c3` isolates the clutter model as the sole
   independent variable. No detector can gain from a different windowing,
   padding or guard convention — a confound that is present in the source
   scripts (see [`BUG_LOG.md`](../BUG_LOG.md) L4, L5, L6, where the four legacy
   scripts implement **two different decision rules** and **two different guard
   geometries** between them).

*Reproduce:* `_common/cfar_compose.m`, `*/`*`CFAR_TLog.m`.

---

**F2 — Both 3-parameter families depend on `c3` only through the scale-free
log-skewness, so their shape tables are 1-D, not 2-D.**

Eliminating the scale between the MoLC equations gives, for generalized gamma,

```
    c3²/c2³  =  ψ₂(k)² / ψ₁(k)³   =:  g(k)
```

and for Burr XII,

```
    c3/c2^{3/2}  =  (ψ₂(1) − ψ₂(κ)) / (ψ₁(κ) + ψ₁(1))^{3/2}   =:  B(κ)
```

Both right-hand sides are functions of the shape parameter alone, and both
left-hand sides are **dimensionless** — invariant to any rescaling of the image.
So `k` and `κ` each depend on a *single scalar*, not on the `(c2, c3)` pair.
Verified strictly monotone over `κ, k ∈ [10⁻⁴, 10⁴]` (200 001-point scan).

**Consequence.** The generalized-gamma and Burr-XII shape ROMs are 1-D — a few
kilobits — rather than 2-D, which would be megabits. This is the single fact
that makes either 3-parameter detector affordable on a Cyclone V. The second
parameter then follows with one square root:

```
    1/v   = sign(−c3)·√c2 / √ψ₁(k)                    (generalized gamma)
    1/ρ   = √c2 / √(ψ₁(κ) + ψ₁(1))                     (Burr XII)
```

Both factorise into a shape-only part and a `c2`-only part, so — exactly as the
combined `K/C` LUT did for the Weibull datapath — **no runtime divider is needed
for either detector.**

G0 in its two-parameter `(L, α)` mode is the **sole exception**: it is a genuine
2-D shape map and does not reduce to one address variable.

*Reproduce:* `GenGammaCFAR_Params.m`, `BurrCFAR_Params.m`, `_common/mono_table_invert.m`.

---

## 2. The supplied generalized-gamma cubic is an asymptotic approximation

**F3 — The cubic in `gengammamolc.m` is identically `r = 8(k+1)²/(2k+1)³`, the
large-`k` asymptotic form of the exact MoLC relation, with up to 20% error at
small `k` and a different domain.**

The legacy coefficients

```
    a0 = 8c3²,  a1 = 4(3c3² − 2c2³),  a2 = 2(3c3² − 8c2³),  a3 = c3² − 8c2³
```

divided through by `c2³` and collected in `r = c3²/c2³` give exactly

```
    r·(2k+1)³ = 8·(k+1)²
```

which is what `ψ₁(k) ≈ 1/k + 1/(2k²)` and `ψ₂(k) ≈ −1/k² − 1/k³` reduce
`g(k) = ψ₂(k)²/ψ₁(k)³` to. Confirmed numerically: the cubic's unique positive
real root satisfies this relation to machine precision at every `r` tested.

| `k` (true) | exact `g(k)` | cubic `r(k)` | error in recovered `k` |
|---|---|---|---|
| 0.2 | 3.489 | 4.198 | — |
| 0.3 | — | — | **+10.4%** |
| 0.8 | — | — | −9.1% |
| 1.5 | — | — | −6.8% |
| 4.0 | — | — | −2.0% |
| 12.0 | — | — | −0.3% |

The two also have **different ranges**:

```
    exact  g(k) → 4  as k → 0⁺        so the solver is valid only for r < 4
    cubic  r(k) → 8  as k → 0⁺        so it "solves" for r < 8
```

For `r ∈ (4, 8)` the cubic returns a root for a system that has **no exact
solution**. Measured on SSDD at `sli=21/guard=15`: `r > 4` on **34.9%** of
windows, `r > 8` on only **4.0%** — so the approximation is masking a real
out-of-support condition on roughly a third of the image.

The legacy validity guard `3c3² ≤ 8c2³` (i.e. `r ≤ 8/3`, corresponding to
`k ≳ 0.4`) is neither of these boundaries.

**Recommendation for the paper:** report the exact relation. It costs nothing in
hardware — either way Phase 3 stores a 1-D table addressed by `r`, and the table
may as well hold the exact values. Both solvers are implemented
(`'Solver','cubic'` reproduces the supplied formula, `'Solver','exact'` inverts
`g(k)`) and the sweep runs both.

*Reproduce:* `verify_models.m` §3b; `GenGammaCFAR_Params.m`.

---

## 3. Support conditions, in closed form

**F4 — Each estimator's support condition is closed-form and can be tested
before any iteration.**

| Detector | Exists iff | Numeric boundary |
|---|---|---|
| Weibull | always (`c2 > 0`) | — |
| Lognormal | always (`c2 > 0`) | — |
| Generalized Gamma | `r = c3²/c2³ < 4` (exact) / `< 8` (cubic) | — |
| G0, `(L,α)` | `\|8c3\| < \|ψ₂(ψ₁⁻¹(4c2))\|` | — |
| G0, `L = 1` | `c2 > ψ₁(1)/4` | `0.41123` |
| Burr XII | `−1.139443 < s < 2`,  `s = c3/c2^{3/2}` | `B(∞) = −1.139443` |

Each boundary was derived from the endpoint limits of the corresponding monotone
residual, then verified to fire exactly where predicted (`verify_models.m` §4).

This replaces the legacy behaviour, which had **no support test at all** and
recorded `fsolve`'s exit flag without ever acting on it
([`BUG_LOG.md`](../BUG_LOG.md) L3).

---

**F5 — Two of the support conditions exclude ordinary clutter, for structural
reasons that no implementation can fix.**

**(a) Weibull clutter lies exactly on Burr XII's boundary.** For pure Weibull
clutter, `c2 = ψ₁(1)/C²` and `c3 = ψ₂(1)/C³`, so

```
    s = ψ₂(1) / ψ₁(1)^{3/2} = −2.404114 / 2.109965 = −1.139443
```

— **identically** `B(∞)`, the lower support limit. This is not a coincidence:
Burr XII degenerates to Weibull as `κ → ∞`, so Weibull *is* the boundary of the
Burr family. The consequence is unavoidable: **any clutter even slightly more
negatively log-skewed than Weibull has no Burr-XII MoLC fit**, and estimator
noise alone will push roughly half the windows of genuinely Weibull clutter out
of support. Burr can only extend the model in one direction.

**(b) Single-look G0 requires `C < 2`.** `c2 > ψ₁(1)/4` is, in the Weibull
parameterisation `C = √(ψ₁(1)/c2)`, exactly `C < 2`. Near-Rayleigh clutter sits
at `C ≈ 2`; SSDD's measured median is `C ≈ 3.65`. Measured invalid fraction:
**93.7–95.2%** of windows across all window sizes.

This eliminates the cheapest G0 hardware variant (a 1-D shape ROM) on this class
of data and forces the 2-D `(L, α)` mode.

*Reproduce:* `verify_models.m` §4; `_comparison/Results/dataset_comparison.csv`.

---

## 4. The dataset finding — and why it changes the conclusion

**F6 — On SSDD, the 3-parameter estimators fail on a large fraction of windows.
That failure is an artefact of the dataset's JPEG encoding, not a property of
SAR clutter.**

Running the *same* front end over both datasets at `sli=21/guard=15`:

| Dataset | median `c2` | median log-skew `s` | % outside Burr support | % outside exact GΓD |
|---|---|---|---|---|
| **MSTAR**, native complex SAR | 0.073 | **−0.413** | **0.4%** | **0.0%** |
| **MSTAR**, rounded to 8-bit | 0.082 | −0.725 | 4.5% | 0.0% |
| **SSDD**, 8-bit JPEG | 0.123 | **−1.420** | **56.1%** | 35.0% |

The physics is identical — X-band SAR amplitude in every row. Only the
**encoding** differs.

**Mechanism.** `x = log(sqrt(I+0.5))` is very steep at the bottom of the 8-bit
range: `I=0 → x=−0.347`, `I=1 → +0.203`, `I=2 → +0.458`, against a typical
clutter level of `x ≈ 2`. A single `I=0` pixel in a 216-cell window is a
~2.3-unit outlier, and `c3` weights outliers by the **cube**. A handful of
near-black pixels dominates the third moment while barely moving the second.

**Quantization alone is not the whole story.** Rounding MSTAR to 8 bits moves
the median skewness from −0.413 to −0.725 (support failure 0.4% → 4.5%) — real
but modest. The much larger remaining gap to SSDD's −1.420 / 56.1% belongs to
**lossy JPEG compression**, whose ringing scatters *isolated* near-black pixels
through otherwise uniform clutter.

**The controlling variable is isolation, not count.** MSTAR contains *more* dark
pixels overall (27.3% vs SSDD's 10.4%) and is barely affected, because those are
large genuinely-dark background regions. A uniform dark area shifts a window's
mean; it does not skew it. Stratifying SSDD windows by local near-black fraction
shows the same thing from the other side — the *most* extreme skew is in windows
with only **1–5%** dark pixels (median `s` = −2.158, 91.6% out of support),
**not** in the darkest windows:

| local near-black fraction | windows | median `s` | % below Burr limit |
|---|---|---|---|
| < 1% | 205 444 | −0.641 | 26.7% |
| 1–5% | 159 133 | **−2.158** | **91.6%** |
| 5–20% | 4 728 | −1.099 | 47.6% |
| 20–50% | 42 152 | −0.641 | 2.9% |
| 50–80% | 17 188 | −0.959 | 27.7% |

### Consequence for the paper

**A Phase 2 ranking computed on SSDD systematically understates Generalized
Gamma and Burr XII.** Any published comparison of 2- vs 3-parameter CFAR models
that uses a JPEG-encoded benchmark is measuring the codec as much as the clutter
model. This is a methodological point worth making in its own right, and it
means the FPGA decision must not rest on SSDD alone.

*Reproduce:* `_comparison/compare_datasets.m`, `_comparison/run_darkpixel_study.m`.

---

**F11 — F6 replicates on a second, independent real-SAR platform: spaceborne
Sentinel-1 GRD (SARFish), C-band, a different sensor, band and processing chain
from MSTAR's airborne X-band.**

MSTAR alone leaves open the possibility that its good behaviour in F6 is a
property of *that* sensor/processor rather than of uncompressed SAR generally.
Nine 1024×1024 native-uint16 clutter crops from one Sentinel-1 GRD scene
(`S1A_IW_GRDH_1SDV_20200226T052146…`, VH polarization, no JPEG or other lossy
step anywhere in the ESA processing chain) extend the same measurement:

| Dataset | median `c2` | median log-skew `s` | % outside Burr support | % outside exact GΓD | % dark px |
|---|---|---|---|---|---|
| MSTAR, native complex | 0.073 | −0.413 | 0.4% | 0.0% | 27.3% |
| MSTAR, rounded to 8-bit | 0.082 | −0.725 | 4.5% | 0.0% | 36.7% |
| **SARFish, native uint16** | **0.022** | **−0.365** | **1.3%** | **0.2%** | **0.6%** |
| SSDD, 8-bit JPEG | 0.123 | −1.420 | 56.1% | 35.0% | 10.4% |

SARFish's skew (−0.365) is the least negative of all four rows — even milder
than MSTAR's own native figure — and its support-violation rates (1.3%, 0.2%)
sit an order of magnitude below SSDD's, on data that shares nothing with MSTAR
except "not lossily compressed."

**This also strengthens F6's specific mechanism, not just its headline.** F6's
claim is that *isolated* near-black pixels from lossy quantization corrupt `c3`,
not dark pixels as such — evidenced there by MSTAR having a *higher* raw dark-
pixel fraction than SSDD (27.3% vs 10.4%) while being far better behaved.
SARFish adds the other end of that same argument: its dark-pixel fraction
(0.6%) is the lowest of any row here, by more than an order of magnitude, and
belongs to a sensor whose 16-bit floor is a real physical noise floor rather
than an 8-bit code value produced by quantization. A platform with the least
compression-like behaviour of any row also has the least corrupted `c3` — the
two ends of F6's mechanism now have data on both sides.

**Caveat.** Nine crops from one scene, one polarization, no ship ground truth
— this corroborates F6's clutter-statistics claim, it does not by itself
establish `Pd`/`Pfa` on spaceborne data (open question 1 is still open for
that reason). It is included as an independent replication of the *mechanism*,
not as a second `Pd` benchmark.

*Reproduce:* `sarfish_sample/download_sample.py`, `sarfish_sample/extract_crops.py`,
`_comparison/compare_datasets.m` (SARFish arm), `Results/dataset_comparison.csv`.

---

**F12 — On HRSID (verified lossless, ship-annotated — the first dataset here
with both properties at once), the detector ranking is not what SSDD showed,
and two detectors reveal a real large-window penalty SSDD's grid was too
coarse to see clearly.**

This directly answers open question 1 below: an annotated non-JPEG dataset
now exists and has been run. `run_comparison_hrsid.m` and `run_comparison.m`
were both run at `NumImages=60` (fixed across every `sli` point — confirmed
by `ShipsTotal` staying constant down every column, 189 for HRSID / 131 for
SSDD) over `sli=[11 15 17 21 31 41 51 61 71 81 91 101 111 121 131 141 151]`,
`Pd` read at matched **measured** `Pfa≈10⁻³` (F7/F9 methodology):

| Detector | HRSID peak `sli` | HRSID peak `Pd` | SSDD peak `sli` | SSDD peak `Pd` |
|---|---|---|---|---|
| Weibull | 61 | **0.979** | 81 | 0.748 |
| Lognormal | 131 (still rising at grid edge) | 0.841 | 81 | 0.603 |
| GenGamma | 141 | 0.778 | 81 | 0.786 |
| G0 | 81 | 0.640 | 61 | 0.397 |
| BurrXII | 91 | 0.720 | 71 | 0.542 |

**The ranking flips.** On SSDD, GenGamma had the highest `Pd` at nearly every
`sli` (answering open question 8's "why does GenGamma win at small sli" —
it turns out this was itself an SSDD-specific effect, not a general one).
On HRSID, **Weibull dominates outright** (0.95–0.98 across `sli=51–151`)
while GenGamma trails at 0.54–0.78. Removing the JPEG artifact did not
unlock 3-parameter superiority — the simplest 2-parameter model wins
cleanly on real, uncompressed SAR ship data. This is a distinct claim from
F6/F11, which only established a *coverage* failure (invalid-window
fraction), not a `Pd`-ranking result.

**G0 and BurrXII show a real, sizeable peak-then-decline on HRSID** that
Weibull/Lognormal/GenGamma do not: G0 falls from 0.640 at `sli=81` to 0.370
by `sli=151` (27 points), BurrXII from 0.720 at `sli=91` to 0.561 (16
points). The other three detectors stay much flatter past their peaks. Read
together with F6's mechanism (G0/BurrXII's shape/tail parameters are the
ones most sensitive to a handful of corrupting pixels), this suggests those
same parameters are also the most sensitive to reference-window homogeneity
breaking down at large window sizes — a second, independent way the same
class of parameter is the fragile one.

**Caveat.** Absolute `Pd` is not comparable between the two datasets (open
question 6 already establishes this generally) — HRSID's higher `Pd`
throughout likely reflects its finer resolution (0.5–3m vs SSDD's 1–10m) and
different ship-size mix, not a "HRSID is easier" claim on its own. The
ranking-flip and peak-sli findings are what's load-bearing here, not the
absolute numbers. Lognormal's HRSID curve had not clearly plateaued by
`sli=151` — treat its `sli=131` entry as provisional pending a wider sweep.

*Reproduce:* `_comparison/run_comparison_hrsid.m`, `_comparison/run_comparison.m`,
`Tag='_hrsid_full'` / `Tag='_ssdd_full_fixed60'`,
`Results/comparison_summary_hrsid_full.csv`,
`Results/comparison_summary_ssdd_full_fixed60.csv`.

---

## 5. The calibration finding

**F7 — Comparing detectors at a common *nominal* `Pfa` is invalid, because they
do not reach the same operating point.**

At nominal `Pfa = 10⁻⁴`, `sli = 21/guard = 15` (smoke-test subset):

| Detector | measured `Pfa` | ratio to nominal | ship `Pd` |
|---|---|---|---|
| Weibull | 3.35e-03 | **33.5×** | 0.857 |
| GenGamma | 4.19e-03 | 41.9× | 0.143 |
| Lognormal | 1.88e-04 | **1.9×** | 0.000 |
| G0 | 2.83e-04 | 2.8× | 0.000 |
| Burr XII | 4.28e-04 | 4.3× | 0.143 |

Reading "Weibull has the best `Pd`" off this table compares a detector running
**33× looser than it claims** against one running nearly on-calibration. The
apparent `Pd` advantage is an operating-point difference, not a discrimination
difference.

**Methodological consequence, adopted throughout this work:**

1. `Pfa` is swept across **six decades** (10⁻¹ … 10⁻⁶) so every detector traces
   its own operating curve.
2. The headline comparison is **`Pd` against *measured* `Pfa`** (a ROC view),
   not against nominal.
3. The qualitative side-by-side figure bisects each detector's nominal `Pfa`
   until its *measured* background rate hits a common target, so every panel
   spends the same false-alarm budget.
4. The measured-vs-nominal ratio is itself reported as a **primary result** — it
   is the most direct evidence of whether a clutter model actually describes the
   data.

*Reproduce:* `run_comparison.m`, `plot_comparison.m` (Figs 2 and 3),
`plot_detection_maps.m`.

---

**F8 — Invalid windows must be excluded from the false-alarm denominator, or a
detector that failed to run looks well-calibrated.**

A window where the estimator has no solution cannot produce a detection.
Counting it as "background correctly rejected" credits the detector for coverage
it never provided. With G0 at ~23% invalid and Burr at up to ~91% on some SSDD
images, this is not a rounding correction — it is the difference between a
meaningful and a meaningless `Pfa`.

`cfar_metrics.m` therefore measures `Pfa` only over valid background, and
reports `CoverageFraction` alongside it. Both `Pd` and `Pfa` must be read
together with coverage.

---

## 6. Verification

Every threshold formula was validated **independently of the others**, by
generating synthetic clutter from each detector's own distribution, estimating
parameters from it exactly as the real detector does, and measuring the actual
exceedance rate against the nominal `Pfa`. An error here would shift a
detector's whole operating point without producing any internal inconsistency —
no amount of comparing the five against each other would reveal it.

| Detector | `Pfa` = 10⁻² | 10⁻³ | 10⁻⁴ |
|---|---|---|---|
| Weibull | 1.00 | 1.00 | 1.05 |
| Lognormal | 1.00 | 0.98 | 1.01 |
| Generalized Gamma | 1.00 | 1.02 | 1.05 |
| G0 | 1.00 | 0.98 | 0.97 |
| Burr XII | 0.99 | 1.00 | 1.00 |

(measured/nominal ratio, 4×10⁶ samples each; sampling noise alone is ~5% at
`Pfa` = 10⁻⁴)

Supporting checks, all passing:

| Check | Result |
|---|---|
| Shared front end vs. the hardware-validated `WeibullCFAR_Floating` | **0 pixels** differ at `sli` = 15/21/31; max relative threshold difference `1.6e-11` |
| Log-domain ≡ amplitude-domain decision, 7 detector configurations | **0 mismatched pixels** |
| Estimator round-trips from known parameters | `1.3e-16` (Weibull) … `2.9e-14` (GΓD, Burr), `6.6e-12` (G0) |
| Support conditions fire at the derived boundaries | pass |
| Burr overflow guard, `κ=0.01`, `Pfa=10⁻⁶` | finite (`δ=1281.57`) vs `Inf` for the legacy form |

The first row doubles as a regression test on the shared front end: if it ever
drifts from the windowing the existing Weibull RTL implements, that is where it
shows up.

*Reproduce:* `_comparison/verify_models.m`.

---

## 7. Efficiency comparison

**Run:** 100 SSDD images (evenly spaced through all 1160) × 7 window
geometries × 7 detector configurations × 6 nominal `Pfa` values = 29 400 grid
cells, 2638 s wall clock. Raw: `Results/comparison_raw_main.csv` (29 400 rows).
Pooled: `Results/comparison_summary_main.csv` (294 rows).

### 7.1 The headline result — F9, tail miscalibration dominates everything else

**F9 — At small nominal `Pfa`, the measured/nominal ratio diverges by 2-3
orders of magnitude for Weibull and Generalized Gamma, while Lognormal stays
within one order of magnitude throughout.** This is the single largest effect
in the whole sweep — larger than any `Pd` difference between detectors — and it
was not visible in the Phase 1 calibration check (§6), which only tested down
to `Pfa = 10⁻⁴` on synthetic, single-distribution clutter.

Measured/nominal `Pfa` ratio, `sli = 21`, pooled over 100 images (**16.5M
background pixels** at the tightest point — 11 356 false detections at
`Pfa=10⁻⁶` for Weibull alone, so this is not sampling noise: Poisson relative
error there is ~1%):

| nominal `Pfa` | Weibull | Lognormal | Gen. Gamma | G0 | Burr XII |
|---|---|---|---|---|---|
| 10⁻¹ | 0.80 | 0.58 | 0.93 | 1.00 | 1.14 |
| 10⁻² | 1.34 | 0.26 | 2.13 | 0.59 | 0.92 |
| 10⁻³ | 4.21 | 0.31 | 9.63 | 0.67 | 1.20 |
| 10⁻⁴ | 18.9 | 0.73 | 58.6 | 1.58 | 3.03 |
| 10⁻⁵ | 108 | 2.93 | 413 | 6.71 | 12.3 |
| **10⁻⁶** | **714** | **19.3** | **3162** | 31.8 | 56.7 |

At `Pfa = 10⁻⁶`, asking Weibull for one false alarm per million background
pixels delivers one per **1 400** (714× too many); asking Generalized Gamma
delivers one per **316** (3162× too many). Lognormal is the only detector whose
worst-case ratio stays under 20×.

**Why the Phase 1 gate did not catch this.** §6's calibration check draws
*i.i.d.* samples directly from each detector's own distribution — by
construction the model is exactly right there, and it confirms only that the
*formula* is implemented correctly. F9 is a **real-data** effect: the achieved
`Pfa` is governed by the *true* clutter tail, which no fitted 2- or
3-parameter model matches at extreme quantiles, and by MoLC estimation noise
on a finite window.

**The mechanism (corrected).** An earlier draft of this section attributed the
difference between detectors to how much each one *amplifies shape-estimate
noise*, describing Weibull's threshold as a power law in `C` and Lognormal's as
additive in `σ`. **That explanation was wrong** and is replaced by the
following, which is exactly verified (see F10):

In the log-amplitude domain both detectors' offsets are *linear in the same
statistic*:

```
    δ_Weibull   = [ K(Pfa) / √ψ₁(1) ] · √c2        K(Pfa) = γ + log(−log Pfa)
    δ_Lognormal = [ z(Pfa)          ] · √c2        z(Pfa) = √2·erfcinv(2·Pfa)
```

They consume the same estimate (`c2`), with the same linearity, so neither
amplifies estimation noise more than the other. The entire difference is in
**how fast the `Pfa`-dependent constant grows as `Pfa → 0`**:

| | asymptotic growth | at `Pfa=10⁻²` | at `10⁻⁶` |
|---|---|---|---|
| Weibull, `K/√ψ₁(1)` | `~ log log(1/Pfa)` | 1.641 | 2.497 |
| Lognormal, `z` | `~ √(2 log(1/Pfa))` | 2.326 | 4.753 |
| ratio | — | 1.418 | 1.903 |

`K` grows like **log-log** — almost flat. Asking Weibull for a `Pfa` ten times
smaller raises its threshold offset by only a few percent, so the achieved
false-alarm rate barely moves and the ratio explodes. Lognormal's `z` grows
like **√log**, fast enough that the achieved rate actually follows the request
much further down. Lognormal is not fitting the clutter better; its `Pfa → δ`
mapping is simply far more conservative, increasingly so as `Pfa` shrinks.

Generalized Gamma is worst of the five (3162× at `10⁻⁶`) because its offset
`[log Γ⁻¹(Pfa,k) − ψ(k)]/v` carries the same slow `Pfa` growth *and* is
divided by `v`, which is derived through `k` — itself an unstable inversion of
the noisy ratio `r = c3²/c2³` (two already-noisy moments, cubed and squared).
Here the noise-amplification argument *does* apply, and it compounds with F3's
finding that the cubic solver is biased at small `k`.

**Consequence for the paper.** *A detector's published `Pd` at a stated nominal
`Pfa` is not comparable across detectors unless the measured/nominal ratio is
reported alongside it* — and F9 shows that ratio is not a fixed property of a
detector, it **grows geometrically as `Pfa` shrinks**, at a different rate per
detector. Any single-point comparison (e.g. "`Pd` at `Pfa=10⁻⁴`") silently picks
a different *actual* false-alarm budget for each detector. This is the
quantitative form of F7's qualitative warning, and it is considerably more
severe than the calibration gap F7 was written to describe — extending it from
"some detectors run loose" to "detectors amplify MoLC noise into tail error at
wildly different rates, and the gap widens without bound as `Pfa` shrinks."

*Reproduce:* `Results/comparison_summary_main.csv`, column `PfaRatio`.

### 7.1b F10 — Weibull and Lognormal are the *same detector* up to one constant

**F10 — In the unclamped region, the Weibull and Lognormal CFAR decision rules
are identical up to a single `Pfa`-dependent scalar. They are not two competing
clutter models on this data; they are one decision rule with two calibrations.**

Both estimate the same statistic and produce offsets proportional to `√c2`:

```
    δ_Lognormal / δ_Weibull  =  z(Pfa)·√ψ₁(1) / K(Pfa)   — independent of c2
```

Measured over a decade range of `c2` at `Pfa = 10⁻³` and `10⁻⁶`, the ratio is
constant to **4×10⁻¹⁶ and 6.7×10⁻¹⁶** respectively — machine precision, i.e.
exact. The two detectors differ *only* through:

1. the scalar above (1.418 at `Pfa=10⁻²` rising to 1.903 at `10⁻⁶`), and
2. their **clamps**, which are the sole source of genuinely different
   behaviour: Weibull saturates at `C=8` for `c2 < ψ₁(1)/64 = 0.0257`, while
   Lognormal's `σ` clamp does not engage until `c2 < 0.0025`. In the band
   `c2 ∈ [0.0025, 0.0257]` — the smoothest, darkest windows — Weibull's
   threshold is pinned flat while Lognormal's keeps tracking the data.

**This explains §7.2 directly.** Weibull and Lognormal come out "essentially
tied" on the matched-measured-`Pfa` ROC not because two different clutter
models happened to perform alike, but because *they trace the same ROC curve*.
Sweeping `Pfa` slides each along that shared curve; matching on measured `Pfa`
then necessarily lands them at the same point, up to the clamp discrepancy.

**Consequences:**

- *For the paper:* a Weibull-vs-Lognormal ROC comparison is close to vacuous
  and should be reported as an equivalence, not a contest. The meaningful
  difference between them is calibration (F9) and clamp behaviour, not
  discrimination. Any published claim that one "outperforms" the other at a
  fixed nominal `Pfa` is measuring the constant in the table above.
- *For hardware:* the two share one datapath and one `√c2` table, differing by
  a single stored constant per `Pfa` — so supporting both costs essentially
  nothing over supporting either. Combined with F2 (GΓD and Burr also need
  `√c2`), **four of the five detectors share the same shape table.**

*Reproduce:* the check in this section is `scratchpad/check_wbl_lgn.m`-style
arithmetic over `WeibullCFAR_TLog` / `LognormalCFAR_TLog`; the constants are
`K(Pfa)/√ψ₁(1)` and `z(Pfa)` as defined in those two files.

**Addendum — the clamp gap is confirmed non-negligible on real data, not just a
theoretical edge case.** A 250-random-image `sli=17` sweep on HRSID (2026-09,
see `_comparison/Results/rand250_report/`) showed Weibull's ship-level `Pd`
far exceeding Lognormal's at matched measured `Pfa≈10⁻³` (0.70 vs 0.29) —
a much larger gap than F10's clamp-band argument alone was expected to
produce. Measuring directly: **4.22% of all `c2` samples at this `sli`/dataset
fall inside `[0.0025, 0.0257)`** — the exact band where Weibull's clamp has
engaged but Lognormal's has not (`scratch_check_clamp.m`-style front-end-only
measurement, 60 images, `sli=17/guard=13`). This confirms the mechanism is
live at a non-trivial rate on real SAR data, not a vanishing corner case.
**Not yet established:** whether ship-adjacent windows are enriched for this
band relative to the whole-image average (plausible if ships tend to sit in
locally smooth/low-variance backgrounds) — this would need a per-ship-window
check, not just the whole-image rate measured so far. Also unexplained by
this mechanism: GenGamma/G0/Burr XII's much larger shortfall behind Weibull
on the same HRSID sweep (0.20/0.30/0.24 vs Weibull's 0.70), despite GenGamma's
`FractionInvalid` being only 0.2% here (i.e., it is not abstaining the way F6
showed on SSDD's JPEG artifact) — a real, currently open question, not
covered by F10's Weibull/Lognormal-specific argument.

### 7.2 ROC — `Pd` at a matched *measured* `Pfa` (the fair ranking)

Because of F9, ranking detectors by nominal `Pfa` is not meaningful. The
correct comparison holds the *measured* background false-alarm rate fixed
across detectors — approximately `10⁻³` here (the exact value found for each
detector differs slightly; it is the nearest point each detector's own sweep
reaches to that target):

| `sli` | Weibull | Lognormal | Gen. Gamma | G0 | Burr XII |
|---|---|---|---|---|---|
| 11 | 0.13 @ 9e-4 | 0.27 @ 3e-3 | **0.29 @ 2e-3** | 0.06 @ 6e-4 | 0.12 @ 1e-3 |
| 15 | 0.24 @ 9e-4 | 0.08 @ 4e-4 | **0.32 @ 2e-3** | 0.13 @ 7e-4 | 0.20 @ 1e-3 |
| 17 | 0.33 @ 8e-4 | 0.08 @ 3e-4 | **0.34 @ 3e-3** | 0.20 @ 7e-4 | 0.25 @ 1e-3 |
| 21 | 0.48 @ 1e-3 | **0.50 @ 3e-3** | 0.32 @ 3e-3 | 0.19 @ 7e-4 | 0.24 @ 1e-3 |
| 31 | 0.65 @ 9e-4 | **0.66 @ 2e-3** | 0.39 @ 4e-3 | 0.25 @ 7e-4 | 0.31 @ 9e-4 |
| 41 | 0.66 @ 9e-4 | **0.68 @ 2e-3** | 0.42 @ 4e-3 | 0.22 @ 8e-4 | 0.35 @ 8e-4 |
| 51 | 0.79 @ 1e-3 | **0.80 @ 2e-3** | 0.52 @ 4e-3 | 0.31 @ 9e-4 | 0.40 @ 9e-4 |

**Weibull and Lognormal are essentially tied and dominate at every window size
`sli ≥ 21`**, with Lognormal marginally ahead from `sli=21` up. **Generalized
Gamma wins only at the smallest windows** (`sli ≤ 17`, where reference-cell
count is lowest) — plausibly because a 3-parameter fit has an advantage when
2-parameter models are most starved of data, an effect worth its own
investigation. **G0 and Burr XII trail throughout the swept range.**

This directly **contradicts** the naive reading of Fig 1 (`Pd` at fixed nominal
`Pfa`), where Weibull appears to lead by a wide margin at every window size —
that reading is an artefact of F9: Weibull's measured `Pfa` at a "matched"
nominal value is systematically far looser than Lognormal's, so Fig 1 is
comparing detectors at different operating points, not different discrimination
ability.

*Reproduce:* Fig 2 (`02_roc_measured.png`); `Results/comparison_summary_main.csv`.

### 7.3 Object-level F1 and the invalid-window confound

Best pooled F1 anywhere in the grid, per detector:

| Detector | best F1 | at | `Pd` there | invalid% there |
|---|---|---|---|---|
| Burr XII | 0.407 | `sli=51, Pfa=10⁻⁶` | 0.21 | 58.0% |
| G0 | 0.364 | `sli=51, Pfa=10⁻⁵` | 0.15 | 77.1% |
| Lognormal | 0.363 | `sli=51, Pfa=10⁻⁴` | 0.24 | 0.0% |
| Weibull | 0.352 | `sli=51, Pfa=10⁻⁶` | 0.72 | 0.0% |
| Gen. Gamma | 0.031 | `sli=51, Pfa=10⁻⁶` | 0.52 | 2.7% |

This table is a caution, not a ranking: Burr's and G0's "best" F1 both occur at
the smallest nominal `Pfa` tested, where F9 has already driven their measured
`Pfa` far from nominal, and at invalid-window fractions of 58-77%. F1 here is
computed only over windows where a detection was *possible* (`cfar_metrics.m`
excludes invalid windows from the background denominator, per F8), so a high
invalid fraction is not directly inflating this number — but it does mean
Burr's and G0's best operating points cover a small, non-random subset of each
image, which §7.2's matched-`Pfa` comparison controls for and this raw
best-F1 table does not. **Use §7.2 for ranking; this table is included only to
show that F1-at-best-nominal-`Pfa` is a misleading summary on its own.**

Generalized Gamma's collapse to F1=0.031 at its "best" nominal point is F9
again: at `Pfa=10⁻⁶` its measured rate is `3162×` too high, so its precision
collapses even though `Pd=0.52` looks respectable.

*Reproduce:* Fig 5 (`05_f1_vs_window.png`).

### 7.4 Computational cost and estimator robustness

Estimator time (ms/megapixel) and invalid-window fraction, `Pfa=10⁻⁴` (front
end excluded — it is identical across detectors and adds ~0.5-3 ms/MP
depending on `sli`):

| `sli` | Weibull | Lognormal | Gen. Gamma | G0 | Burr XII |
|---|---|---|---|---|---|
| 11 | 20.5 ms, 0% | 14.9 ms, 0% | 291 ms, 6.8% | 5773 ms, 75.2% | 599 ms, 43.2% |
| 21 | 18.6 ms, 0% | 15.1 ms, 0% | 297 ms, 4.6% | 5770 ms, 78.7% | 522 ms, 54.0% |
| 51 | 18.4 ms, 0% | 14.7 ms, 0% | 294 ms, 2.7% | 5600 ms, 77.1% | 497 ms, 58.0% |

Four observations:

1. **Lognormal is cheapest and Weibull nearly ties it** — both are closed-form,
   no iteration.
2. **G0 is ~300× more expensive than Weibull/Lognormal**, even after the
   tabulated-`Q` optimisation in [`BUG_LOG.md`](../BUG_LOG.md) D2 (which itself
   gave a 5× speedup). This is a MATLAB vectorised-bisection cost, not
   representative of hardware — the RTL equivalent is a handful of fixed-depth
   pipeline stages, not 30 iterations — but it is a real cost for running this
   sweep and would be a real cost for an on-chip *iterative* solver, which is
   exactly why Phase 3 replaces this with a ROM.
3. **Burr's invalid fraction rises with `sli`** (43% → 58%) while
   **Generalized Gamma's falls** (6.8% → 2.7%). Larger windows give a less
   noisy `c3` estimate, which should help both — and does help GΓD — but Burr's
   support band `(−1.139, 2)` sits far closer to where real clutter's skewness
   concentrates (F5a: Weibull clutter sits *exactly* on Burr's lower boundary),
   so a *less* noisy estimate more often lands cleanly on the wrong side of that
   nearby boundary rather than being pulled back into range. This is a second,
   independent way F5a's structural point shows up in the data.
4. **G0's invalid fraction is nearly flat with `sli`** (75% → 79% → 77%), unlike
   Burr and GΓD. This is consistent with F5b/H4: G0's dominant failure mode on
   this data is the `C ≈ 3.65` median clutter shape sitting well outside its
   support band in a way that more reference cells cannot fix, because the
   problem is the clutter's *location* relative to the support boundary, not
   noise around it.

*Reproduce:* Fig 4 (`04_cost_and_validity.png`).

### 7.5 Shape-parameter spread

Median and inter-quartile range of each detector's primary shape parameter,
`sli=21, Pfa=10⁻³`, pooled over images — this sets how finely the Phase 3 ROM
must resolve the shape axis, and (per §7.1) how much MoLC noise the tail
formula will amplify:

| Detector | parameter | median | IQR | IQR / median |
|---|---|---|---|---|
| Lognormal | `σ` | 0.337 | 0.069 | **0.20** |
| Weibull | `C` | 3.804 | 0.730 | 0.19 |
| Burr XII | `κ` | 2.874 | 3.841 | 1.34 |
| Gen. Gamma | `k` | 0.504 | 1.349 | **2.68** |
| G0 | `u=−α` | 13.84 | 30.43 | **2.20** |

Lognormal's and Weibull's shape estimates are tight relative to their median
(~20%); the three-parameter detectors and G0 are far noisier (relative IQR
130-270%). This ranks the detectors in the *same order* as F9's tail-noise
amplification, which is the expected relationship: a noisier shape estimate
feeding a power-law tail formula is exactly the mechanism F9 identifies.

*Reproduce:* `Results/comparison_summary_main.csv`, columns `MedianShape`,
`IQRShape`.

### 7.6 Solver variants

`sli=21, Pfa=10⁻⁴`:

| Variant | `Pd` | F1 | invalid% | Pfa ratio |
|---|---|---|---|---|
| GenGamma (cubic, supplied formula) | 0.403 | 0.019 | 4.6% | 58.6 |
| GenGamma-exact | 0.399 | **0.031** | 32.4% | 40.7 |
| G0 (`L,α`, both estimated) | 0.088 | 0.118 | 78.7% | 1.6 |
| G0-L1 (single-look, `L=1`) | 0.076 | **0.139** | 94.7% | 4.5 |

**GenGamma-exact has better F1 and a better-calibrated Pfa ratio than the
cubic**, at the cost of a much higher invalid fraction (32% vs 5%) — the exact
solver is more often correctly *refusing* to answer rather than answering with
F3's biased value. Given Phase 3 stores either as the same size 1-D ROM, this
is a genuine argument for shipping the exact relation.

**G0-L1's invalid fraction (94.7%) makes it unusable on this dataset** despite
a marginally higher F1 than the full `(L,α)` mode — that F1 is computed over
the ~5% of windows it *can* evaluate, which per F5b are systematically the
least representative (lowest-`C`, most heterogeneous) windows in the image, not
a random sample.

*Reproduce:* Fig 6 (`06_variants.png`).

### 7.7 Qualitative comparison at matched false-alarm budget

Fig 8 matches all five detectors to the same **measured** background `Pfa`
(bisected on nominal `Pfa` per detector) on one 3-ship scene (`000139.jpg`,
`sli=21/guard=15`, target `Pfa=10⁻³`):

| Detector | nominal Pfa used | measured Pfa achieved | `Pd` | FP objects | invalid% |
|---|---|---|---|---|---|
| Weibull | 4.0e-9 | 1.00e-3 | 1.000 | 65 | 0.0% |
| Lognormal | 2.8e-3 | 1.00e-3 | 1.000 | 62 | 0.0% |
| Gen. Gamma | ≤1e-12 (bracket floor) | 1.81e-3 | 0.000 | 105 | 0.1% |
| G0 | 5.1e-4 | 1.09e-3 | 0.000 | 8 | 90.7% |
| Burr XII | 1.1e-4 | 1.01e-3 | 0.000 | 56 | 37.7% |

Two things stand out. First, Weibull and Lognormal find all three ships at this
budget and the other three find none — consistent with the §7.2 ranking on this
window size. Second, **Generalized Gamma could not be pushed down to the
`10⁻³` target at all**: even at the bisection's floor (nominal `Pfa=10⁻¹²`) its
measured rate stayed at `1.81e-3`. This is F9 made concrete on a single image —
at `sli=21` its calibration ratio is already `9.6×` at nominal `10⁻³` and grows
without bound as nominal `Pfa→0`, so there is a **hard floor** on how tight a
background rate it can be made to hit, no matter how conservative the nominal
value requested.

*Reproduce:* Fig 8 (`08_detection_maps.png`); `plot_detection_maps.m`.

### 7.8 Does the `IFloor` fix (F6) actually help?

`run_darkpixel_study.m`, `sli=21/guard=15, Pfa=10⁻²`, 60 images, IFloor swept
`{0,1,2,4}`:

| Detector | IFloor=0 | IFloor=1 | IFloor=2 | IFloor=4 |
|---|---|---|---|---|
| Burr XII, invalid% | 53.5% | 46.1% | 35.7% | **15.9%** |
| Burr XII, F1 | 0.039 | 0.035 | 0.024 | 0.017 |
| Gen. Gamma, invalid% | 4.1% | 1.1% | 1.0% | 5.1% |
| Gen. Gamma, F1 | 0.017 | 0.013 | 0.013 | 0.015 |
| G0, invalid% | 79.1% | 74.3% | 78.5% | 78.4% |
| Weibull, F1 | 0.033 | 0.027 | 0.025 | 0.022 |
| Lognormal, `Pd` | 0.527 | 0.611 | 0.649 | **0.702** |

**`IFloor` does exactly what F6 predicted for Burr XII's support failure**
(53.5% → 15.9% invalid at `IFloor=4`, a 3.4× reduction) but **does essentially
nothing for G0** (79.1% → 78.4%), confirming §7.4's point #4: G0's failure mode
is not the dark-pixel artefact at all, it is the clutter's shape sitting
outside `L=1`-adjacent support regardless of a few outlier pixels. Generalized
Gamma's invalid fraction is non-monotone in `IFloor` (4.1% → 1.1% → 1.0% →
5.1%) — clamping helps up to a point and then starts moving otherwise-valid
windows' skewness in the wrong direction.

**F1 does not improve for any of the two 3-parameter detectors** despite the
invalid-fraction win for Burr — being back in support is not the same as being
well-fitted (§9 open question 2 is answered: no, at least not on this metric,
at this `Pfa`). **Lognormal's `Pd` improves substantially and monotonically**
(0.527 → 0.702) even though it has no support condition to fix — `IFloor`
simply removes an outlier that was dragging `σ` up, tightening its threshold.
This is an unplanned but genuine secondary benefit, and since `IFloor` is free
in hardware (§4, F6), **`IFloor=4` looks worth adopting for Lognormal
specifically**, independent of its effect on the 3-parameter detectors.

*Reproduce:* Fig 7 (`07_darkpixel_study.png`); `Results/darkpixel_study.csv`.

### Window geometries swept

`sli = 17` is included specifically to bracket the hardware's `SLI = 18`, the
largest window that fits Cyclone V under default optimization —
`cfar_front_end` requires an odd window, so 18 cannot be evaluated directly.
`sli = 51` is the software-optimal point established by the existing Weibull
project. The gap between them is the accuracy price of the current device.

| `sli` | 11 | 15 | 17 | 21 | 31 | 41 | 51 |
|---|---|---|---|---|---|---|---|
| `guard` | 7 | 11 | 13 | 15 | 21 | 27 | 41 |
| `N` cells | 72 | 104 | 120 | 216 | 520 | 952 | 920 |

Guard sits at 0.65–0.80 of `sli` because SSDD ship boxes have median
max-dimension ~37 px; a smaller guard lets a ship's own bright pixels leak into
its reference window.

---

## 8. Hardware implications (established so far)

| Finding | Implication |
|---|---|
| **F1** | One shared front end + a swappable back-end ROM, not five accelerators |
| **F2** | GΓD and Burr need **1-D** shape ROMs (kilobits), and **no runtime divider** — both reciprocals factorise into a shape-only and a `c2`-only part |
| **F2** | G0 `(L,α)` is the only genuine 2-D shape map of the five |
| **F3** | Use the exact GΓD relation, not the cubic — same ROM, better values |
| **F5b** | The cheap single-look G0 (1-D ROM) is unusable on this class of data |
| **F6 / F11** | Do not select detectors for silicon on JPEG-encoded benchmark data — replicated on a second, independent spaceborne platform (F11), not just MSTAR |
| **F12** | On real (non-JPEG) ship-annotated data, Weibull has the highest `Pd` of all five detectors, not GenGamma — the per-detector `SLI` target for 6.2b's ZCU104 synthesis should come from HRSID's peak-`sli` table, not SSDD's; G0/BurrXII specifically should not be synthesized larger than their measured peak (`sli=81`/`91`), since `Pd` actively falls past that point |
| **F9** | **The Q-format precision behind `C` (Weibull) and `k`/`v` (GΓD) is not a rounding detail — it sets how many orders of magnitude the achieved `Pfa` drifts from nominal at small `Pfa`.** A ROM quantisation step that is small relative to §7.5's measured shape-estimate *noise* adds negligible extra error; the noise floor, not the ROM's own resolution, is what F9 shows dominating at small `Pfa`. This argues for sizing the shape ROM from the noise floor rather than from a target quantisation error alone. |
| **F9** | Lognormal's structurally additive tail (§7.1) makes it the most *robust* choice at low `Pfa`, not merely the cheapest — a genuine argument for it as the production default if Phase 4's DSP budget turns out tight, independent of Phase 2's `Pd` ranking |
| — | Lognormal's back end is one LUT + one constant multiply, **no reciprocal at all** — the cheapest of the five by a wide margin |
| **F13** | K-distribution's shape is a 1-D address (`c2` alone, like Weibull/Lognormal) despite needing a numerically-built (not closed-form) delta table — plausibly a cheap backend once quantized into a ROM, but this is a software-structure expectation, not a confirmed synthesis result (Phase 4b not started) |

Inherited, measured, from the existing Weibull DE10-Standard bring-up:

- `SLI = 18 / GUARD = 11` is the Cyclone V ceiling under default optimization —
  51% ALM, **73% DSP**, 8% memory.
- **DSP is the binding resource** — and that is for a *two*-moment detector.
  Adding `c3` adds a third accumulator chain. This is the critical-path risk in
  the whole plan; Phase 4 opens with a synthesis probe of a `window_sum`
  carrying a `c3` chain alone, before any full datapath is written.

---

## 9. Open questions

Explicitly **not** established. Do not cite these as results.

1. ~~Do the 3-parameter detectors win on native SAR data?~~ **Answered by
   F12: no.** HRSID (verified lossless, ship-annotated) gives the first
   `Pd`/`Pfa` measurement on non-JPEG data, and Weibull — the simplest
   2-parameter model — dominates outright (0.95–0.98 vs GenGamma's 0.54–0.78
   at the same `sli` range). Removing the JPEG artifact did not reveal
   3-parameter superiority.
2. ~~Does `IFloor` recover the lost performance, or only the support fraction?~~
   **Answered by §7.8: only the support fraction, for the 3-parameter
   detectors.** Burr's invalid rate falls 3.4× but its F1 does not improve;
   being back in support is not the same as being well fitted. Unexpectedly,
   Lognormal's `Pd` improves substantially (0.53→0.70) even though it has no
   support condition — open sub-question: does that gain hold outside
   `sli=21, Pfa=10⁻²`, and is it specific to SSDD's JPEG artefact (F6) or would
   it also help on clean data?
3. **`c3` fixed-point dynamic range.** The centring constant removes the
   mean-induced cancellation but not the intrinsic range.
4. **Whether a third moment fits Cyclone V at all** at a useful `SLI`.
5. **Whether G0's 2-D shape map is affordable**, or whether G0 is ruled out of
   the DE10 phase by memory.
6. **Absolute `Pd` numbers here are not comparable to the existing Weibull
   project's 88.7%.** That figure used `sli=51/guard=41` on all 1160 images with
   a different decision rule (see F7 and [`BUG_LOG.md`](../BUG_LOG.md) L4). Only
   *within*-sweep comparisons in §7 are valid.
7. **What drives F9's amplification quantitatively?** §7.1 gives a mechanistic
   argument (power-law vs. additive tail) but does not fit or verify a model of
   *how* the ratio scales with `(−log Pfa)` per detector — e.g. whether
   Weibull's ratio is closer to linear or a higher power in `(−log Pfa)`. Worth
   a dedicated derivation and fit for the paper, since it would let the ratio be
   predicted rather than only measured.
8. ~~Why does Generalized Gamma win only at small `sli`~~ **Answered by F12:
   it was SSDD-specific, not general.** On HRSID, GenGamma does not win at
   any `sli` in `[11,151]` — Weibull wins throughout. The apparent
   small-`sli` GenGamma advantage in §7.2 does not replicate on non-JPEG
   data.
9. **K-distribution's SLI-plateau and hardware cost are unstudied — and now
   there's a real question of whether it's viable at all.** F13's `sli=17/51`
   real-data check found K's support condition is met by only ~6% (SSDD) /
   ~0.05% (HRSID) of windows — a near-total collapse. Whether a MUCH larger
   `sli` (more reference samples → less noisy `c2` estimates, possibly
   pushing more windows above the `pi^2/24` threshold) recovers usable
   coverage, or whether single-look (`L=1`) K-distribution is simply not
   viable on this class of data at any practical window size, is open and
   worth checking before investing in Phase 4b's fixed-point/RTL work.

---

## 10. Reproducing everything

```matlab
cd F:\Projects\CFAR
cfar_setup();

verify_models                 % §6  — the verification gate
compare_datasets              % §4  — F6/F11, the encoding finding
                               %      (F11's SARFish arm needs
                               %      sarfish_sample/download_sample.py then
                               %      extract_crops.py run once first)
run_comparison('NumImages',100,'Pfa',[1e-1 1e-2 1e-3 1e-4 1e-5 1e-6],'Tag','_main');
plot_comparison('Tag','_main');   % §7 figures
plot_detection_maps();            % matched-Pfa qualitative comparison
run_darkpixel_study();            % §4 / open question 2
```

| Output | Contents |
|---|---|
| `Results/comparison_raw_main.csv` | one row per (image, geometry, detector, `Pfa`) |
| `Results/comparison_summary_main.csv` | pooled per (detector, geometry, `Pfa`) |
| `Results/dataset_comparison.csv` | F6 |
| `Results/darkpixel_study.csv` | `IFloor` sweep |
| `Figures/01…08` | PNG + vector PDF |

`Pd` is **pooled** (total ships detected / total ships), never averaged over
images — SSDD contains many single-ship images and a per-image mean over-weights
them.

---

## 11. K-distribution (added 2026-09)

**F13 — K-distribution fits this project's unified decision rule (F1) with a
single-scalar (1-D) shape address like Weibull/Lognormal, but needs a
numerically-built threshold instead of a closed form.**

K-distributed amplitude (single-look, `L=1`, matching every detector here)
arises from the classic texture mixture: intensity `I | tau ~
Exponential(mean tau)`, `tau ~ Gamma(a, mu/a)`, amplitude `V = sqrt(I)`. Its
log-cumulants follow directly from that mixture (derived two independent
ways — the mixture representation directly, and cross-checked against
`CFAR K/legacy/kmolc.m`'s pre-existing formula, which agree exactly):

```
    c1 = kappa1(a, mu) = 0.5*(psi(a) - log(a) - EulerGamma) + 0.5*log(mu)
    c2 = ( psi(1,a) + psi(1,1) ) / 4        (psi(1,1) = pi^2/6, exact)
```

so the shape parameter `a` inverts via the **inverse trigamma function** —
reusing `_common/inv_trigamma.m` (already relied on by G0), not a fresh
solver. **Support condition:** a finite positive `a` exists only when
`c2 > pi^2/24` (~0.4112) — numerically **identical** to G0-L1's own support
boundary, not a coincidence: both reduce to the same `psi(1,shape) = 4c2 -
psi(1,1)` form once `L=1` is fixed.

**Unlike every other detector here, K has no elementary quantile function.**
G0 reduces exactly to an F-distribution (incomplete beta), Burr XII has a
closed CDF, GenGamma reduces to the regularized incomplete gamma — K's
density involves a modified Bessel function of the second kind, with no
closed-form CDF. Rather than evaluate that Bessel function directly
(numerically delicate at Pfa=10⁻⁶), the threshold offset `delta(a, Pfa)` is
built from the SAME mixture representation as a 1-D numerical integral with
no special functions beyond the Gamma density:

```
    S(x; a, mu=1) = P(I > x) = E_tau[ exp(-x/tau) ]
                  = integral_0^inf  exp(-x/tau) * gampdf(tau; a, 1/a) dtau
```

solved for `x` at `S(x)=Pfa` via `fzero`, then converted to the log-amplitude
offset. This is well-conditioned across the full Pfa range this project
sweeps (10⁻³ … 10⁻⁶), and is cached as a 161-point grid per Pfa (only 4
distinct Pfa values are ever swept), so the expensive part runs at most 4
times per MATLAB session rather than once per pixel.

**Verification** (`_comparison/verify_models.m`, sections 3/4/5): round-trip
recovers a known shape `a` to max relative error 4.3×10⁻¹⁵ (essentially
machine precision); calibration on synthetic K clutter lands within
0.98–1.04× nominal Pfa across 10⁻²/10⁻³/10⁻⁴ — the same rigor bar the other
five detectors were held to, not a lighter one.

**Not the same pipeline as the legacy K script.**
`CFAR K/legacy/Main_Estimation_and_Detection_K.m` runs Gamma-MAP texture
filtering first, then a separate Gamma-texture CFAR stage (`nkgmolc.m`) on
the filtered image — a two-stage pipeline with no equivalent in this
project's shared single-pass front end, and not comparable to the other five
detectors (which all see raw single-look log-amplitude data, no despeckling
stage). `KCFAR_Floating.m` instead applies the true K-distribution MoLC and
quantile relations directly to the same `cfar_front_end` moments every other
detector uses, so a six-way comparison stays apples-to-apples.

**This is not just this project's own legacy script — it's the published
group's own architectural choice too (BC-4, resolved 2026-09).** REF6
(Mahapatra et al., *IEEE Access* 2026, "An Efficient CFAR Detector for Burr
Type-XII...") states directly that their own CFAR-K comparator requires
"Γ-MAP estimation followed by CFAR detection in Γ distributed background
texture," and its Table 3 shows a "MAP Estimation" step that applies ONLY to
their CFAR-K entry, not to CFAR-LGN/CFAR-WBL/CFAR-G0. REF7 independently
confirms the underlying reason a single-stage route is hard: K's CDF
"precludes a closed-form detection threshold" because of its Bessel-function
term. So the near-total collapse measured above is not merely this
implementation being unlucky — it's consistent with why the published
literature on this exact clutter model always reaches for a despeckling
stage first, rather than applying K's own MoLC/quantile directly to raw
data the way this project's other five (six) detectors do. Full reasoning:
`PAPER3_DRAFT_multimodel-hardware.md` §3.3.

**Consequence for hardware (Phase 4b, not yet built):** K's shape comes from
`c2` alone — a 1-D address space like Weibull's and Lognormal's, not a 2-D
pair like G0's `(L,u)`. Once the delta grid above is quantized into a ROM,
the backend should look structurally like Weibull's/Lognormal's cheap
1-address-ROM design, not G0's/Burr's more expensive one. This is a
plausible expectation from the software structure, **not yet confirmed by
real synthesis** — do not cite it as a resource number until
`quartus_map`/`quartus_fit` actually says so.

**On real data, K is close to unusable at `sli=17/51` — a genuine negative
result, not a bug.** The 250-random-image SSDD/HRSID comparison
(`_comparison/Results/rand250_report/`) measured `FractionInvalid` directly:
~94% on SSDD, **~99.95% on HRSID**, at `sli=17` — meaning K's support
condition (`c2 > pi^2/24`) is met by only a tiny fraction of real windows.
`Pd` correspondingly collapses (SSDD 0.076→0.14 across `sli`=17/51 at
nominal `Pfa=10⁻³`; HRSID 0.001→0.008 — barely above zero). Unlike every
other detector in this comparison, **K is WORSE on HRSID than on SSDD** —
the opposite of the general JPEG-artifact pattern (F6/F12) — because
HRSID's finer resolution (0.5–3m vs SSDD's 1–10m) produces smoother,
lower-local-variance clutter, pushing `c2` below K's threshold even more
often. This is the same failure mode this project already documented for
G0's single-look variant (F5b, G0-L1) — both single-look (`L=1`) reductions
of a texture-mixture model turn out to need more variance than typical real
single-look SAR clutter windows actually have. `verify_models.m`'s round-trip
and calibration checks still pass cleanly (the math is correct); K's
estimator and threshold simply have almost nothing to work with in this
regime.

**Tried the obvious fix — classical Gamma-MAP despeckling — and it makes the
collapse WORSE, not better, which is mechanistically the correct answer, not
a surprise in hindsight.** `CFAR K/gamma_map_filter.m` implements the
classical single-look Lopes Gamma-MAP despeckling filter (vectorised from
`CFAR K/legacy/Main_Estimation_and_Detection_K.m`'s formula, window=7),
applied as image preprocessing before `cfar_front_end` — same K-CFAR math
(`KCFAR_Params`/`KCFAR_TLog`, unchanged) as the plain-K row above, isolating
despeckling as the only variable (`CFAR K/run_comparison_kmap.m`, detector
name `K-MAP7`, same 250-image/seed=42 samples as the plain-K sweep). Result:
on HRSID, `Pd` goes from already-near-zero to **exactly 0 at every Pfa and
both `sli`**, with `FractionInvalid` rising from ~99.95% to ~99.98%. On SSDD
it's closer to a wash (small gain at `sli`=17, small loss at `sli`=51, no
clear direction). **This makes sense once stated plainly:** Gamma-MAP
filtering is a smoothing operation by design — it exists to *reduce* local
variance and suppress speckle — while K's support condition needs local
variance (`c2`) to be *large enough* to clear the `pi^2/24` threshold.
Despeckling pushes `c2` in exactly the wrong direction for this specific
failure mode. Deliberately did NOT also swap in the legacy script's
different post-filter estimator (`nkgmolc.m`) here — changing two things at
once would have left it unclear whether despeckling or the different
estimator caused any change; this result isolates despeckling alone.
(One real bug caught and fixed while building the despeckling filter itself:
the MAP quadratic formula is analytically always non-negative, but produced
spurious small negative "texture" values from floating-point cancellation
when the local shape estimate got very small — fixed by flooring the shape
estimate the same way `KCFAR_Params.m` already does, `AMin=0.05`, plus a
defensive `max(quad,0)`. Not written up as a separate BUG_LOG entry since it
never reached a saved result — caught by the pre-sweep smoke test.)

*Reproduce:* `CFAR K/KCFAR_Params.m`, `CFAR K/KCFAR_TLog.m`,
`CFAR K/KCFAR_Floating.m`, `CFAR K/gamma_map_filter.m`,
`CFAR K/run_comparison_kmap.m`, `_comparison/verify_models.m` (K's blocks in
sections 3/4/5); the real-data results above are
`Results/comparison_summary_{ssdd,hrsid}_rand250_k.csv` (plain K),
`Results/comparison_summary_kmap7_{ssdd,hrsid}.csv` (K-MAP7), and
`Results/rand250_report/SSDD_vs_HRSID_sli17_51_report.docx`.

**Then tried the ACTUAL literature pipeline — Gamma-MAP despeckle followed by
a genuine Gamma-distribution CFAR, not K re-applied to the despeckled image —
and it fully rescues the collapse, decisively answering "does the two-stage
architecture work" with yes.** `K-MAP7` (above) held the detector fixed and
only changed the input, which isolated despeckling as a variable but is not
the pipeline REF6/REF7 (BC-4) or this project's own legacy script actually
describe: both despeckle *and* swap the second-stage distribution to plain
Gamma, because after despeckling the residual texture is no longer a
K-mixture — the multiplicative speckle factor that produced the mixture is
gone, so modelling it as plain single-look Gamma is the theoretically correct
choice, not an arbitrary substitution. New files `CFAR K/GammaTexCFAR_Params.m` /
`GammaTexCFAR_TLog.m` implement exactly this: shape `a = inv_trigamma(4*c2)`
(vs. K's `inv_trigamma(4*c2 - pi^2/6)` — the speckle term is simply absent),
so the support condition drops from K's real, frequently-unmet floor
(`c2 > pi^2/24`) to just `c2 > 0`, which `cfar_front_end`'s own `C2Floor`
already guarantees unconditionally. The threshold is exact closed form via
`gammaincinv` (no `fzero`, no grid cache) — `delta = 0.5*(log(gammaincinv(1-Pfa,a)) - psi(a))`
(one real bug caught and fixed here, BUG_LOG D24: the first version omitted
the `1/a` scale correction between `Gamma(a,1)`'s quantile and the texture's
actual `Gamma(a,1/a)` scale — invisible at `a=1` where the missing factor is
1, but wildly wrong elsewhere; caught by a calibration test, not by the
round-trip check).

Full 250-image sweeps (`Detector='GammaTex-MAP7'`, same seed=42 samples as
every other rand250 row, `Results/comparison_summary_gammatex7_{ssdd,hrsid}.csv`):

| Dataset | sli | Pfa | K (single-stage) Pd | K-MAP7 Pd | **GammaTex-MAP7 Pd** | GammaTex-MAP7 FractionInvalid | GammaTex-MAP7 PfaRatio |
|---|---|---|---|---|---|---|---|
| SSDD  | 17 | 1e-3 | 0.141 | 0.159 | **0.503** | 0% | 3.85x |
| SSDD  | 51 | 1e-3 | 0.265 | 0.254 | **0.880** | 0% | 2.02x |
| HRSID | 17 | 1e-3 | 0.001 | 0.000 | **0.741** | 0% | 0.97x |
| HRSID | 51 | 1e-3 | 0.008 | 0.003 | **0.917** | 0% | 0.83x |

`FractionInvalid = 0` at **every** `sli`/`Pfa` combination on both datasets —
exactly as the support-condition analysis predicted, since `c2 > 0` is
essentially unconditional. On HRSID this is the single most dramatic result
in the whole comparison: single-stage K is within noise of completely
non-functional (`Pd`≈0.1–0.8%, invalid on ~99.96% of windows), and the
two-stage pipeline turns it into the *best-calibrated* detector in the
project (`PfaRatio` 0.83–0.97x, i.e. slightly conservative, at every Pfa
tested) with `Pd` up to 0.92. On SSDD the rescue is just as complete in
`Pd`/`FractionInvalid` terms, though `PfaRatio` runs 2–4x over nominal at the
tightest Pfa values (1e-3 down to 1e-6) — comparable in order of magnitude to
plain K's own SSDD over-calibration (1.7–260x across the same grid, see the
summary CSVs), so the `Pd` gain is not an artifact of drastically loosened
calibration, but the SSDD arm's calibration is looser than HRSID's and worth
tightening (larger `sli`, or a Pfa-correction table) before treating the SSDD
numbers as production-ready.

**Bottom line for BC-4 / Phase 4b:** the published group's own architectural
choice (despeckle, then a *different*, simpler second-stage distribution) is
not merely defensible — on this project's own data it is the difference
between K being unusable and K-family detection being the best result on
HRSID. Any future hardware work on the K route should target this two-stage
GammaTex architecture, not single-stage K.

*Reproduce (GammaTex-MAP7):* `CFAR K/GammaTexCFAR_Params.m`,
`CFAR K/GammaTexCFAR_TLog.m`, `CFAR K/gamma_map_filter.m` (shared with
K-MAP7), `CFAR K/run_comparison_kmap.m` (`'Stage2','GammaTex'`);
`Results/comparison_summary_gammatex7_{ssdd,hrsid}.csv`.
