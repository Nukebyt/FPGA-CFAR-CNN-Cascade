// Builds PAPER2_Explainer_contrast_peak_and_pooling.docx : plain-language explanation, mathematics, and the exact code of
// (i) 2x2 log-mean pooling and (ii) contrast-peak events.  Code snippets are read from the source files at build time (exact text).
// usage: NODE_PATH=<dir with node_modules/docx> node build_explainer_docx.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, HeadingLevel,
  BorderStyle, LevelFormat, Footer, PageNumber } = require("docx");

const HERE = __dirname;
const FIG = path.resolve(HERE, "..", "Results", "pd_study", "explain");
const FIG2 = path.resolve(HERE, "..", "Results", "pd_study", "report");
const OUT = path.resolve(HERE, "..", "..", "PAPER2_Explainer_contrast_peak_and_pooling.docx");
const FONT = "Calibri", MONO = "Consolas";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const P = (parts, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...o, children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 140 }, children: [new TextRun({ text: t, font: FONT, size: 30, bold: true, color: "1F3864" })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 }, children: [new TextRun({ text: t, font: FONT, size: 25, bold: true, color: "2F5496" })] });
const B = (t) => run(t, { bold: true });
const I = (t) => run(t, { italics: true });
const eq = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 }, children: [new TextRun({ text: t, font: "Cambria Math", size: 23 })] });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 }, children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
const callout = (title, text) => new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360], rows: [new TableRow({ children: [new TableCell({
  width: { size: 9360, type: WidthType.DXA }, shading: { fill: "EAF3E6", type: ShadingType.CLEAR, color: "auto" }, margins: { top: 100, bottom: 100, left: 160, right: 160 },
  borders: { top: { style: BorderStyle.SINGLE, size: 4, color: "8DB87A" }, bottom: { style: BorderStyle.SINGLE, size: 4, color: "8DB87A" }, left: { style: BorderStyle.SINGLE, size: 18, color: "5B8F3E" }, right: { style: BorderStyle.SINGLE, size: 4, color: "8DB87A" } },
  children: [new Paragraph({ spacing: { after: 60 }, children: [run(title, { bold: true })] }), new Paragraph({ spacing: { after: 0, line: 276 }, children: [run(text)] })] })] })] });

function img(dir, file, wpx, caption) {
  const buf = fs.readFileSync(path.join(dir, file)); const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); const s = wpx / w;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(w * s), height: Math.round(h * s) }, altText: { title: file, description: caption.slice(0, 150), name: file } })] }),
  new Paragraph({ spacing: { after: 200 }, children: [run(caption, { size: 19, italics: true })] })];
}
// exact source text between two marker substrings (inclusive), from a project file
function snippet(file, startMarker, endMarker) {
  const lines = fs.readFileSync(path.resolve(HERE, file), "utf8").split(/\r?\n/);
  const a = lines.findIndex((l) => l.includes(startMarker));
  if (a < 0) throw new Error("marker not found: " + startMarker);
  let b = a; if (endMarker) { b = lines.findIndex((l, i) => i >= a && l.includes(endMarker)); if (b < 0) throw new Error("end marker not found: " + endMarker); }
  return lines.slice(a, b + 1);
}
const code = (lines, label) => [
  new Paragraph({ spacing: { before: 100, after: 20 }, keepNext: true, children: [run(label, { size: 18, italics: true, color: "555555" })] }),
  ...lines.map((l) => new Paragraph({ spacing: { after: 0, line: 240 }, shading: { fill: "F3F3F3", type: ShadingType.CLEAR, color: "auto" }, indent: { left: 120 },
    children: [new TextRun({ text: l.length ? l : " ", font: MONO, size: 16 })] })),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
];
const SRC = "extract_cnn_patches_pooldet.m";
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" }; const borders = { top: border, bottom: border, left: border, right: border };
function table(cols, header, rows) {
  const cell = (t, i, hdr) => new TableCell({ width: { size: cols[i], type: WidthType.DXA }, borders, shading: hdr ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 }, children: [new Paragraph({ children: [new TextRun({ text: String(t), font: FONT, size: 19, bold: hdr })] })] });
  return new Table({ width: { size: cols.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: cols,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }), ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))] });
}
const stats = JSON.parse(fs.readFileSync(path.join(FIG, "pooling_stats.json"), "utf8"));
const c = [];

// ------------------------------------------------------------------------------------------------ title
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Contrast-peak events and 2×2 log-mean pooling", font: FONT, size: 40, bold: true, color: "1F3864" })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run("How the two key changes to the Weibull-CFAR prescreen work: in plain words, in mathematics, and in code", { italics: true, size: 24 })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [run("Paper 2 working note, 3 October 2026", { size: 20 })] }));
c.push(callout("The whole idea in three sentences", "The old detector measured “normal sea brightness” in a thin ring around each pixel, so a big bright ship ended up inside its own ring and hid itself. Shrinking the picture (averaging 2×2 blocks of the log brightness) makes the ship half as big relative to the ring and calms the speckle, so ships stand out about 1.75 times more. Then, instead of marking the first pixel of every bright blob, we mark the brightest-contrast spot of each bright patch, so the mark stays on the ship even when blobs merge."));
c.push(P(""));

// ------------------------------------------------------------------------------------------------ PART A
c.push(H1("Part A. In simple terms"));
c.push(H2("A.1 What the prescreen is for"));
c.push(P("A radar picture of the sea has hundreds of thousands of pixels. A small neural network (the CNN) is good at deciding “ship or not” but too slow to look at every pixel. So a cheap first stage, the prescreen, scans the picture and puts a circle (we call it an event) on every spot that might be a ship. The CNN then looks only inside the circles."));
c.push(P([B("The one rule: "), "if the prescreen never circles a ship, the CNN can never find it. So the prescreen must circle almost every real ship, even if it also circles some sea speckle by mistake (the CNN removes those)."]));
c.push(H2("A.2 Why the old prescreen lost ships"));
c.push(P("For every pixel the old prescreen asked: “is this pixel brighter than the normal sea around it?” To measure the normal sea it looked at a ring of pixels around the pixel, but the ring was only 2 pixels thick (a 17×17 square with a 13×13 hole). A ship wider than the hole pokes into the ring. The detector then thinks the “normal sea” is bright, and the ship no longer stands out against its own background. We measured this: 9.4 of the 15.7 percentage points of lost ships were ships hiding in their own ring."));
c.push(H2("A.3 Change 1: 2×2 log-mean pooling"));
c.push(P([B("What it does. "), "Radar brightness is turned into a logarithm (so that the grainy speckle becomes an additive wobble rather than a multiplicative one). Then every 2×2 square of four pixels is replaced by the average of its four log-brightness values. An 800×800 picture becomes 400×400."]));
c.push(...img(FIG, "fig_pooling.png", 430, "Figure 1. Pooling: each 2×2 block of pixels becomes one pixel whose value is the mean of the four log-brightness values."));
c.push(P([B("Why it helps, in three ways:")], { keepNext: true }));
c.push(bullet([B("The ship looks half as big. "), "The ring is still 17×17 (now pooled) pixels, but one pooled pixel is two original pixels wide, so the ring now covers twice as much real area. A ship up to about 26 original pixels across (instead of 13) fits inside the hole and no longer contaminates its own ring."]));
c.push(bullet([B("The sea gets calmer, the ship does not. "), "Averaging four noisy readings cancels part of the noise, but a ship is bright over many pixels, so its average stays bright. We measured on 60 images: the local noise variance falls " + stats.var_ratio_median.toFixed(1) + " times (noise s.d. about " + Math.sqrt(stats.var_ratio_median).toFixed(2) + " times smaller). The ship therefore stands out about " + Math.sqrt(stats.var_ratio_median).toFixed(2) + " times more clearly against the noise."]));
c.push(bullet([B("Four times fewer pixels. "), "There are 4× fewer places for noise to fake a ship and, in hardware, 4× fewer pixels to process with half-width line buffers."]));
c.push(P([B("What it costs. "), "Detail: ships only a few pixels across get blurred, and positions are accurate to about 2 original pixels. (The CNN’s window is much larger than that, so this does not matter for the second stage.)"]));
c.push(...img(FIG, "fig_schematic.png", 620, "Figure 2. (a) Where events are placed along one image row. (b) Averaging four samples halves the noise but leaves the ship’s average unchanged, so the ship separates from the sea."));
c.push(H2("A.4 Change 2: contrast-peak events"));
c.push(P([B("Contrast "), "is simply how much brighter a pixel is than the normal sea around it: contrast = pixel brightness − normal sea brightness (both in log units). Sea speckle has contrast around zero; a ship pixel has high contrast."]));
c.push(P([B("The question. "), "After the detector decides which pixels are bright enough, neighbouring pixels light up together in blobs. Where exactly should we put the circle?"]));
c.push(bullet([B("Old rule (“first pixel of each blob”). "), "Put one circle at the top-left pixel of every blob. It is cheap to compute while the image streams past, but the top-left pixel can be on a faint edge, and when sensitivity is raised neighbouring blobs merge into one big blob that gets only one circle for several ships."]));
c.push(bullet([B("New rule (“contrast peak”). "), "Look at the contrast of the bright pixels like a landscape of hills. Put a circle on every hilltop: a pixel whose contrast is the highest within its 5×5 neighbourhood. A ship’s bright core is a hilltop, merged blobs still have one hilltop per ship, and flat speckle rarely makes a sharp, high hill."]));
c.push(P("A large ship may get a few circles along its bright parts. That is fine: the CNN only needs one circle on the ship to find it."));
c.push(H2("A.5 A real example"));
c.push(...img(FIG, "fig_example.png", 640, "Figure 3. A real HRSID ship (area 642 px, test split). (a) Image with the ship outline. (b) Old design: the ship fills its own reference ring, almost no pixel on it is detected and the events (yellow) land on sea speckle: the ship is missed. (c) New design: the pooled image detects the ship body and the contrast-peak events (green) sit on it. Both designs use Pfa 1e-3 and gate 0.75."));
c.push(H2("A.6 What each change bought (measured on the 842 test images, 2,877 ships)"));
c.push(table([2600, 2200, 2400, 2160], ["Change", "Ships circled (Pd)", "Circles per image", "Main effect"], [
  ["Before (old design, Pfa 1e-2, first-pixel events)", "0.948", "298", "reference"],
  ["+ contrast-peak events", "0.977", "383", "+2.9 points: circles land on ships"],
  ["+ 2×2 log-mean pooling", "0.991", "96", "+1.4 points and 4× fewer circles"],
]));
c.push(P(""));

// ------------------------------------------------------------------------------------------------ PART B
c.push(H1("Part B. The mathematics"));
c.push(H2("B.1 The CFAR detector (what pooling and peaks are applied to)"));
c.push(P(["Let ", I("I"), " be the 8-bit pixel value and ", I("x"), " = ln √(", I("I"), " + ½) the log amplitude. If the clutter amplitude is Weibull, P(A > t) = exp(−(t/B)^C) with shape C and scale B, then ", I("x"), " has a Gumbel-type law with"]));
c.push(eq("E[x] = ln B − γ/C,     Var[x] = π² / (6 C²),     γ = 0.5772…"));
c.push(P(["The detector estimates the two log-cumulants from a reference ring ℛ around the cell under test (sli×sli minus guard×guard, N = sli² − guard² cells, 120 for 17/13):"]));
c.push(eq("c₁ = (1/N) Σ_ℛ x,      c₂ = (1/(N−1)) Σ_ℛ (x − c₁)²"));
c.push(P(["The shape follows in closed form, Ĉ = √(ψ(1,1)/c₂) = π/√(6 c₂) (clamped to [0.8, 8]), and the scale from c₁. Setting the false-alarm probability P(A > T) = P", new TextRun({ text: "fa", subScript: true, font: FONT, size: 22 }), " gives the threshold in the log domain:"]));
c.push(eq("T = c₁ + δ,     δ = K(P_fa)/Ĉ = (√6/π) · K(P_fa) · √c₂ ≈ 0.78 · K · σ̂,     K(P_fa) = γ + ln(−ln P_fa)"));
c.push(table([2340, 2340, 2340, 2340], ["P_fa", "K", "δ / σ̂ = 0.78 K", "ρ* (see B.3)"], [
  ["3×10⁻²", "1.832", "1.43", "0.329"], ["10⁻²", "2.104", "1.64", "0.271"], ["10⁻³", "2.510", "1.96", "0.207"], ["10⁻⁴", "2.798", "2.18", "0.174"],
]));
c.push(P(""));
c.push(P(["A pixel is ", B("detected"), " if x > T. Its ", B("contrast"), " is C(p) = x(p) − c₁(p) ≥ 0 for detected pixels, and the ", B("gate"), " keeps those with C ≥ τ (0.75 in the baseline, 0.6 in the final design). The ring sums are computed with integral images, ", I("S"), "(ℛ) = Box", new TextRun({ text: "sli", subScript: true, font: FONT, size: 22 }), " − Box", new TextRun({ text: "guard", subScript: true, font: FONT, size: 22 }), ", so the cost is O(1) per pixel regardless of window size."]));
c.push(H2("B.2 2×2 log-mean pooling"));
c.push(P("Definition. With f = 2 the pooled image is"));
c.push(eq("x_p(i, j) = ¼ Σ_{a,b ∈ {0,1}} x(2i + a, 2j + b)"));
c.push(P(["(the mean of the ", I("logs"), ", i.e. the log of the geometric mean of the four amplitudes; the alternative “intensity-mean” pools I before the log and behaved similarly). All of B.1 is then applied unchanged on the 400×400 grid, so a 17/13 ring spans 34×34 / 26×26 original pixels and is 4 original pixels thick."]));
c.push(P(["Noise reduction. For four background samples of variance σ² with neighbour correlations ρ", new TextRun({ text: "h", subScript: true, font: FONT, size: 22 }), " (horizontal), ρ", new TextRun({ text: "v", subScript: true, font: FONT, size: 22 }), " (vertical) and ρ", new TextRun({ text: "d", subScript: true, font: FONT, size: 22 }), " (diagonal), the six pairs in a block give"]));
c.push(eq("Var(x_p) = (σ² / 4) · (1 + ρ_h + ρ_v + ρ_d)"));
c.push(P(["Independent samples would give the factor 4. Measured on 60 HRSID images (median local 17×17 variance of x, full resolution over pooled): ratio " + stats.var_ratio_median.toFixed(2) + " (10–90 %: " + stats.var_ratio_p10.toFixed(2) + "–" + stats.var_ratio_p90.toFixed(2) + "), with lag-1 correlation ρ ≈ " + stats.lag1_corr_median.toFixed(2) + " in the data (speckle is slightly oversampled). With ρ", new TextRun({ text: "h", subScript: true, font: FONT, size: 22 }), " = ρ", new TextRun({ text: "v", subScript: true, font: FONT, size: 22 }), " = 0.19 and a small diagonal term this predicts a ratio of about 2.8, consistent with the measurement. Because δ ∝ σ̂, the threshold offset falls by √" + stats.var_ratio_median.toFixed(2) + " ≈ " + Math.sqrt(stats.var_ratio_median).toFixed(2) + ", while the contrast of a ship that is at least one block wide is unchanged: Δ", new TextRun({ text: "p", subScript: true, font: FONT, size: 22 }), " ≈ Δ. The detection condition Δ > δ becomes"]));
c.push(eq("Δ / σ  >  0.78 K / 1.75      (instead of  Δ / σ > 0.78 K )"));
c.push(P("so at P_fa = 10⁻³ a ship needs a contrast of 1.12 σ instead of 1.96 σ (σ the single-look noise s.d.)."));
c.push(H2("B.3 Why a ship that fills its own ring cannot be detected (self-masking)"));
c.push(P(["Let a fraction ρ of the reference cells lie on the same ship, which is brighter than the sea by Δ = x̄", new TextRun({ text: "ship", subScript: true, font: FONT, size: 22 }), " − μ", new TextRun({ text: "bg", subScript: true, font: FONT, size: 22 }), ", and the rest on sea of variance σ². The ring is a two-component mixture:"]));
c.push(eq("c₁′ = μ_bg + ρΔ,      c₂′ ≈ σ² + ρ(1 − ρ)Δ²"));
c.push(P("A ship pixel (log brightness μ_bg + Δ) is detected only if"));
c.push(eq("Δ > ρΔ + 0.78 K √(σ² + ρ(1 − ρ)Δ²)     ⇔     Δ² [ (1 − ρ)² − (0.78 K)² ρ (1 − ρ) ]  >  (0.78 K)² σ²"));
c.push(P(["The bracket must be positive, i.e. (1 − ρ) > (0.78 K)² ρ, which gives"]));
c.push(eq("ρ  <  ρ*  =  1 / (1 + (0.78 K)²)"));
c.push(P(["This is a remarkable property: if more than ρ* of the ring lies on the ship, the pixel cannot be detected ", I("however bright the ship is"), ", because the variance estimate grows with Δ² as fast as the mean offset. ρ* = 0.207 at P", new TextRun({ text: "fa", subScript: true, font: FONT, size: 22 }), " = 10⁻³ (table above) and the measured median ring fraction on the ship for the ships we lost to self-masking was ρ = 0.27 > ρ*. Two remedies follow directly from the formula: raise P", new TextRun({ text: "fa", subScript: true, font: FONT, size: 22 }), " (ρ* = 0.27 at 10⁻², 0.33 at 3×10⁻²) and reduce ρ by pooling (the same ring covers twice the area, so only ships larger than twice the guard still contaminate it). This is exactly the combination in the final design."]));
c.push(H2("B.4 Contrast-peak events"));
c.push(P("Let D be the set of detected pixels, C the contrast field and G = {p ∈ D : C(p) ≥ τ} the gated set. The old rule (streaming NMS corner) is"));
c.push(eq("E_nms = { p ∈ D : p−e_x ∉ D, p−e_x−e_y ∉ D, p−e_y ∉ D, p−e_y+e_x ∉ D } ∩ {C ≥ τ}"));
c.push(P("i.e. the first pixel, in raster order, of each run of detected pixels. Its position depends only on the shape of D, not on the contrast, and the number of events is at most the number of connected components of D (merged blobs give one corner, wherever it falls). The new rule is"));
c.push(eq("E_pk = { p ∈ G : C(p) ≥ C(q)  for all q ∈ G with |q − p|_∞ ≤ 2 }"));
c.push(P("(a 5×5 local maximum of the contrast restricted to the gated pixels; ties are kept, so equal-contrast plateaus on 8-bit data give adjacent duplicate events, which are harmless). Properties:"));
c.push(bullet("Every connected component of G contains its global maximum, so it yields at least one event: |E_pk| ≥ #components(G). A ship whose brightest pixels pass the threshold and the gate is therefore always circled."));
c.push(bullet("The location is determined by the contrast field, so it stays on the bright core when the threshold is lowered and blobs merge (E_nms collapses to one corner per merged blob)."));
c.push(bullet("Spacing: two events with different contrast cannot lie within 2 pixels of each other (the lower one would not be a 5×5 maximum); only exact ties can coexist. On average the new rule produced 181 events per image over the whole data set."));
c.push(P(["Event position in original coordinates (f is the pooling factor, pooled indices 1-based): y", new TextRun({ text: "c", subScript: true, font: FONT, size: 22 }), " = (e", new TextRun({ text: "y", subScript: true, font: FONT, size: 22 }), " − 1) f + (f + 1)/2, likewise x", new TextRun({ text: "c", subScript: true, font: FONT, size: 22 }), ". The CNN window is cut from the 2×2-pooled code image at j = ⌊(y", new TextRun({ text: "c", subScript: true, font: FONT, size: 22 }), " − 1)/2⌋, i = ⌊(x", new TextRun({ text: "c", subScript: true, font: FONT, size: 22 }), " − 1)/2⌋ (32×32, edge-replicated)."]));
c.push(H2("B.5 Cost in hardware (estimates, not yet synthesised)"));
c.push(bullet("Pooling: the cascade already computes the 2×2 mean of the 8-bit log codes for its pooled frame store; the Weibull core then runs on a 400-wide stream: line buffers of (SLI − 1) × 400 instead of × 800 words (half the 118 M10K of the 800-wide core), and a pixel rate four times lower."));
c.push(bullet("Peak events: a 5×5 maximum is separable (maximum over 5 pixels along the row, then over 5 rows), so it needs four extra line buffers of the contrast value (about 4 × 400 × 16 bit ≈ 25.6 kbit) and a 5-input comparator tree, plus a latency of two rows; the NMS rule needed only one bit per pixel from the previous row."));
c.push(bullet("The Weibull LUTs (K(P_fa)/C, the shape grid) were built for the full-resolution c₂ range; with pooling c₂ is about 3× smaller, so the c₂ axis of the LUT has to be re-derived and the fixed-point formats re-checked."));

// ------------------------------------------------------------------------------------------------ PART C
c.push(H1("Part C. The exact code"));
c.push(P(["The MATLAB below is the reference implementation that produced every number in the reports (", I("_comparison/pd_study/extract_cnn_patches_pooldet.m"), " and the equivalent variant harness ", I("pd_variants_eval2.m"), "). The text is copied from the source files at build time. Variable names: ", I("I"), " image, ", I("f"), " pooling factor, ", I("x"), " log amplitude (pooled), ", I("c1, c2"), " ring statistics, ", I("D"), " detections, ", I("C"), " contrast, ", I("Dg"), " gated detections, ", I("E"), " events."]));
c.push(H2("C.1 2×2 log-mean pooling"));
c.push(...code(snippet(SRC, "function B = blockmean", "end"), "blockmean: mean of every f×f block (reshape to f × h/f × f × w/f, average over the two block dimensions)"));
c.push(...code(snippet(SRC, "if f == 1, Id = I; else", "x = blockmean(log(sqrt(I + 0.5)), f)"), "Building the pooled log amplitude x (Mode 'log': mean of the log; otherwise log of the mean intensity)"));
c.push(H2("C.2 Reference-ring statistics with integral images"));
c.push(...code(snippet(SRC, "function [c1, c2] = ring_stats", "function T = nms_trigger").slice(0, -1), "c1 and c2 over the ring (sli×sli minus guard×guard), symmetric padding so the border is evaluated; boxsum_valid is an integral-image box sum"));
c.push(H2("C.3 Weibull shape and threshold offset"));
c.push(...code(snippet("..\\..\\CFAR_Weibull\\WeibullCFAR_Params.m", "PSI11 = pi^2 / 6", "C     = min(max(C_raw"), "WeibullCFAR_Params.m: shape from c2 (method of log-cumulants)"));
c.push(...code(snippet("..\\..\\CFAR_Weibull\\WeibullCFAR_TLog.m", "EULER_GAMMA = 0.5772156649015329", "delta = K ./ prm.C"), "WeibullCFAR_TLog.m: delta = K / C with K = γ + ln(−ln Pfa)"));
c.push(H2("C.4 Detection, contrast, gate and the events"));
c.push(...code(snippet(SRC, "[c1, c2] = ring_stats(x, Gp, TK, TG, X0);", "case 'peak5'"), "Detection D, contrast C, gate Dg and the two event rules (nms = old, peak5 = contrast-peak events)"));
c.push(P([I("Reading the peak5 line: "), "Cd is the contrast where the pixel is gated and −∞ elsewhere; imdilate(Cd, ones(5)) is the 5×5 maximum of Cd; a gated pixel is an event when its own contrast is at least that maximum, i.e. it is the local hilltop."]));
c.push(...code(snippet(SRC, "function T = nms_trigger", "T = D & ~prvL"), "The old event rule (first pixel of each run): a detected pixel with no detected neighbour to the left, upper-left, above or upper-right"));
c.push(H2("C.5 From event to CNN window"));
c.push(...code(snippet(SRC, "[ey, ex] = find(E);", "ry = min(max(round(yc)"), "Event coordinates in original (full-resolution) pixels"));
c.push(...code(snippet(SRC, "qc = floor(min(max((0.5*log(I + 0.5)", "pat(:,:,n) = uint8(P(rr, cc2));"), "The CNN window: 32×32 of the pooled 2×2 code image around the event, edge replicated"));
c.push(H2("C.6 The same prescreen in Python (cross-check)"));
c.push(P(["The figures in this note were drawn with an independent numpy re-implementation (", I("_comparison/pd_study/explain_demo.py"), "); it reproduces the MATLAB logic, including the 5×5 peak rule via a maximum filter."]));
c.push(...code(snippet("explain_demo.py", "def prescreen(I, pool, mode", "return x, c1, c2, D, E, contrast"), "explain_demo.py: prescreen()"));
c.push(H1("Summary table"));
c.push(table([2300, 3500, 3560], ["", "2×2 log-mean pooling", "Contrast-peak events"], [
  ["Applied to", "the image, before the log-CFAR", "the detected+gated pixels, after the CFAR"],
  ["Formula", "x_p = ¼ Σ x over a 2×2 block", "E = { p ∈ G : C(p) ≥ C(q), |q−p|∞ ≤ 2 }"],
  ["Fixes", "self-masking (ρ falls), speckle (σ falls 1.75×), load (4× fewer pixels)", "events off the ship / merged blobs (E_nms collapses)"],
  ["Costs", "resolution (ships < ~4 px blurred, ±2 px position)", "4 contrast line buffers + 5×5 comparator in hardware"],
  ["Measured (test)", "Pd 0.977 → 0.991 and 383 → 96 events/img", "Pd 0.948 → 0.977 at +85 events/img"],
]));

const doc = new Document({
  creator: "Paper 2 working note", title: "Contrast-peak events and 2x2 log-mean pooling",
  styles: { default: { document: { run: { font: FONT, size: 22 } } }, paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, font: FONT }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } } ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, right: 1440, bottom: 1300, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 18 })] })] }) }, children: c }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
