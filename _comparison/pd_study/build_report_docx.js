// Builds PAPER2_Weibull_prescreen_Pd_report.docx (scientific-literature style) from the figures in Results/pd_study/report.
// usage: NODE_PATH=<dir with node_modules/docx> node build_report_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, BorderStyle, LevelFormat, Footer, PageNumber, TabStopType,
} = require("docx");

const FIG = path.resolve(__dirname, "..", "Results", "pd_study", "report");
const OUT = path.resolve(__dirname, "..", "..", "PAPER2_Weibull_prescreen_Pd_report.docx");
const FONT = "Times New Roman";
const CONTENT_W = 9360; // US Letter, 1 inch margins (DXA)

const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const P = (parts, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o,
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: t, font: FONT, size: 26, bold: true })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, children: [new TextRun({ text: t, font: FONT, size: 23, bold: true, italics: true })] });
const I = (t) => run(t, { italics: true });
const B = (t) => run(t, { bold: true });
const SUB = (t) => run(t, { subScript: true });
const SUP = (t) => run(t, { superScript: true });
const eq = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 }, children: [new TextRun({ text: t, font: "Cambria Math", size: 22, italics: true })] });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 }, alignment: AlignmentType.JUSTIFIED,
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });

function img(file, wpx, caption) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);          // PNG IHDR
  const scale = wpx / w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
        altText: { title: file, description: caption.slice(0, 200), name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [run(caption, { size: 19 })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(cols, header, rows, opts = {}) {
  const total = cols.reduce((a, b) => a + b, 0);
  const cell = (t, i, hdr, bold) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA }, borders,
    shading: hdr ? { fill: "E8EDF5", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : (opts.center ? AlignmentType.CENTER : AlignmentType.RIGHT),
      children: [new TextRun({ text: String(t), font: FONT, size: 19, bold: hdr || bold })] })],
  });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: cols,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, opts.boldLast && r === rows[rows.length - 1])) }))] });
}
const caption = (t) => new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true, children: [run(t, { size: 20, bold: false })] });
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

const children = [];
// ------------------------------------------------------------------ title block
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "Raising the Detection Probability of a Weibull-CFAR Ship Prescreen on HRSID: Loss Attribution and a Pooled-Domain Redesign", font: FONT, size: 34, bold: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run("Working draft for Paper 2 (Weibull-CFAR prescreen + CNN discriminator on a Cyclone V SoC)", { italics: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [run("Software study, 3 October 2026. Authors: [to be completed]", { size: 20 })] }));

// ------------------------------------------------------------------ abstract
children.push(H1("Abstract"));
children.push(P([
  "Cascaded SAR ship detectors that use a constant-false-alarm-rate (CFAR) prescreen followed by a convolutional discriminator are bounded by the ship detection probability ",
  I("P"), SUB("d"), " of the prescreen, because the discriminator can only remove candidates. On the HRSID data set (5,604 images, 16,951 ships) our hardware-faithful Weibull-CFAR prescreen (17×17 reference window, 13×13 guard, ",
  I("P"), SUB("fa"), " = 10", SUP("−3"), ") reports ", I("P"), SUB("d"), " ≈ 0.89 under the customary box-hit criterion, but only 0.843 when a ship counts as detected only if the prescreen emits a candidate event on the ship. "
  + "We attribute the 15.7 percentage points lost on a per-ship basis using a clean-sea counterfactual and find that 9.4 points (60 % of the loss) are caused by the ship's own pixels contaminating the 2-pixel-wide reference ring. "
  + "Guided by this attribution we searched 2,400 prescreen variants on a tuning set and confirmed the best on 842 untouched test images. Detecting on the 2×2 log-mean pooled image with a wider reference window, a higher false-alarm rate with a contrast gate, a padded border, and contrast-peak events raises the strict ",
  I("P"), SUB("d"), " from 0.867 to 0.9986 on the test split while reducing the candidate load from 274 to 190 events per image. Over the entire data set the prescreen delivers an event for 99.61 % of ships (66 missed; 99.09 % of images complete); the residual misses are almost exclusively inshore (0.9922 versus 0.9998 offshore). "
  + "The results are obtained in floating point on pooled data; bit-exact hardware validation of the pooled-domain prescreen remains future work.",
]));
children.push(P([B("Keywords: "), "SAR ship detection; CFAR; Weibull clutter; reference-window masking; multi-look pooling; HRSID; FPGA."], { alignment: AlignmentType.LEFT }));

// ------------------------------------------------------------------ 1 Introduction
children.push(H1("1. Introduction"));
children.push(P("Two-stage ship detectors first nominate candidate locations with a statistical detector and then classify each candidate with a compact neural network. This architecture suits low-power platforms because the expensive network runs only on a small fraction of the image [1,2]. Its end-to-end detection probability, however, is the product of the prescreen detection probability and the discriminator's retention of true ships. Any ship missed by the prescreen is irrecoverable, so the prescreen should operate close to unit " + "detection probability and leave false-alarm suppression to the second stage."));
children.push(P("A classical CFAR prescreen estimates local clutter statistics from a reference ring around the cell under test and compares the cell with a threshold derived from the target false-alarm probability. Two failure mechanisms are well known: masking of weaker targets when interfering targets or clutter edges enter the reference cells, and excess false alarms at clutter transitions [3,4]. Censoring schemes that discard bright reference cells have been proposed to mitigate masking [5,6]. What is rarely reported is how much of the observed miss rate of a given implementation is due to each mechanism, because ground-truth evaluation usually uses a lenient box-overlap criterion that hides the problem."));
children.push(P("This work (i) defines hit criteria that cannot be satisfied by sea clutter alone, (ii) attributes, ship by ship, the Pd lost by an existing hardware-faithful Weibull-CFAR prescreen to specific causes, (iii) uses the attribution to select a redesigned prescreen from a systematic search, and (iv) verifies the result on the whole data set, including per-image and per-scene behaviour. The prescreen studied here is the one implemented on a DE10-Standard FPGA in our earlier work; the study is performed in software, using a bit-exact model of the existing hardware as the baseline."));

// ------------------------------------------------------------------ 2 Data & baseline
children.push(H1("2. Data and baseline prescreen"));
children.push(H2("2.1 Data"));
children.push(P(["HRSID [7] comprises 5,604 images of 800×800 pixels with 16,951 annotated ships, each with a bounding box and a polygon mask; inshore and offshore image lists are provided (1,031 and 4,573 images, respectively). Images were split 70/15/15 % by image with a fixed random seed (3,922/840/842 images; 11,855/2,219/2,877 ships). The Weibull prescreen has no trained parameters, so diagnostics use all images; every ",
  I("selection"), " decision of the redesign (windows, thresholds, gates, event rules) was made on 600 randomly drawn training/validation images (1,644 ships) and reported on the 842 test images."]));
children.push(H2("2.2 Baseline: the hardware-faithful Weibull prescreen"));
children.push(P(["The baseline operates on the log-amplitude ", I("x"), " = ln √(", I("I"), " + 0.5) of the 8-bit pixel ", I("I"), ". For every pixel it computes the mean ", I("c"), SUB("1"), " and unbiased variance ", I("c"), SUB("2"), " of ", I("x"), " over a reference ring (17×17 window minus a 13×13 guard, i.e. 120 cells, 2 pixels thick). The Weibull shape ",
  I("C"), " is obtained in closed form from ", I("c"), SUB("2"), " (the second log-cumulant of a Weibull amplitude; see [9] for clutter models), and the log-domain threshold is"]));
children.push(eq("T = c₁ + δ(c₂, Pfa),   δ = [γ + ln(−ln Pfa)] / C,   γ = 0.5772…"));
children.push(P(["A pixel is detected if ", I("x"), " > ", I("T"), ". The outer 8 pixels of the image are not evaluated because the window does not fit. Detected pixels are converted to candidate events by a streaming non-maximum-suppression (NMS) trigger — the first pixel of a detection run with no detected neighbour to the left or above — followed by a contrast gate ",
  I("x"), " − ", I("c"), SUB("1"), " ≥ 0.75 that removes weak triggers. The hardware model uses fixed-point arithmetic and lookup tables and has been verified bit-exact against the RTL; the floating-point model is used where arbitrary ", I("P"), SUB("fa"), " must be evaluated."]));
children.push(...img("fig_pipeline.png", 600, "Figure 1. Baseline prescreen (top) and the proposed prescreen, configuration A (bottom). The proposed chain detects on the 2×2 log-mean pooled image with a wider, padded reference ring, a higher false-alarm rate with a lower contrast gate, and emits contrast-peak events."));

// ------------------------------------------------------------------ 3 Methods
children.push(H1("3. Methods"));
children.push(H2("3.1 Hit criteria and performance measures"));
children.push(P("Four per-ship hit criteria were evaluated on the same detections:"));
children.push(bullet([B("Box hit: "), "at least one detected pixel inside the ground-truth rectangle (the criterion used by the standard evaluation code)."]));
children.push(bullet([B("Mask hit: "), "at least one detected pixel on the ship polygon."]));
children.push(bullet([B("Component hit: "), "a gated trigger exists in a detection component that touches the box (the label rule used to train the discriminator)."]));
children.push(bullet([B("Event hit (strict, used throughout): "), "a gated candidate event lies within 4 full-resolution pixels of the ship polygon (touching ships share events). This is the event the second stage actually receives for the ship, and it cannot be satisfied by an unrelated clutter detection far from the ship."]));
children.push(P(["The box criterion is lenient: at ", I("P"), SUB("fa"), " = 0.1, 10 % of all pixels are flagged and every box ‘hits’ (",
  I("P"), SUB("d"), " = 1.000), whereas only 0.84 of ships receive an event on the ship (Section 4.1). We additionally report the candidate events per image (the load of the second stage), false events (events farther than 4 px from every ship), and the pixel false-alarm rate. Binomial proportions are given with Wilson 95 % intervals."]));
children.push(H2("3.2 Per-ship loss attribution"));
children.push(P(["For each ship we extracted, from both the bit-exact and the floating-point model, the margin ", I("x"), SUB("max"), " − ", I("T"), " of its brightest valid pixel at each ", I("P"), SUB("fa"), ", its geometry (area, distance to the image border and to the nearest ship), and the local sea statistics ",
  "(mean μ", SUB("bg"), " and variance of ", I("x"), " in a 20-pixel ring around the box, excluding all ships dilated by 3 px). A ", I("clean-sea counterfactual"), " replaces the contaminated window statistics by the ship-free ones: ",
  "δ", SUB("clean"), " = δ(var(", I("x"), SUB("bg"), "), 10", SUP("−3"), ") and margin", SUB("clean"), " = ", I("x"), SUB("max"), " − (μ", SUB("bg"), " + δ", SUB("clean"), "). A positive clean-sea margin means that a CFAR fed with ship-free statistics would have detected the ship. "
  + "At the ship's best pixel the 120 reference cells were classified as lying on the ship's own polygon, on a neighbouring ship's polygon, or on non-ship pixels brighter than μ", SUB("bg"), " + 2σ", SUB("bg"), " (bright clutter); the dominant class (≥ 5 % of the ring) names the contamination."]));
children.push(P("Ships lost by the hardware prescreen (no event on the ship) were assigned to mutually exclusive classes in this order: (A) no valid pixel; (B) found by the floating-point model but not by the hardware model; (C) ship pixels detected but no gated event on the ship; (D) not detected although the clean-sea margin is positive, subdivided by ring contamination; (E) not detected and clean-sea margin non-positive (near miss within 0.15 nat or deep)."));
children.push(H2("3.3 Design-space search"));
children.push(P("A single evaluation harness computed, for every image, the strict hit rate and event load of each combination of the following design choices:"));
children.push(bullet([B("Domain: "), "full resolution, or 2×2 / 4×4 pooling, with the pooled value being the mean intensity (multi-look) or the mean of the log amplitude (the quantity held by the pooled frame store of the cascade)."]));
children.push(bullet([B("Reference window: "), "(window, guard) from 9/7 to 49/33 pixels in the evaluated domain."]));
children.push(bullet([B("Border: "), "outer pixels unevaluated (as in the RTL) versus symmetric padding by the half window."]));
children.push(bullet([B("Reference estimator: "), "plain ring versus two-pass iterative censoring (detections of a first pass at P", SUB("fa"), " = 3×10", SUP("−3"), ", dilated 5×5, are excluded from the ring in a second pass)."]));
children.push(bullet([B("Threshold and gate: "), "P", SUB("fa"), " ∈ {10", SUP("−4"), ", 3×10", SUP("−4"), ", 10", SUP("−3"), ", 3×10", SUP("−3"), ", 10", SUP("−2"), ", 3×10", SUP("−2"), "}; contrast gate ∈ {0.5, 0.6, 0.75, 0.9, 1.0} nat."]));
children.push(bullet([B("Event definition: "), "NMS corner (baseline); contrast peak (a gated detected pixel whose contrast is maximal in its 3×3 or 5×5 neighbourhood); or one event per 8-connected component at its highest-contrast gated pixel."]));
children.push(P("The resulting 2,400 variants (plus 612 variants of an earlier run that also contained the censored estimators) were ranked on the tuning images by event-hit Pd at a given event load; shortlisted configurations were then evaluated once on the test images. Event tolerance was fixed at 4 full-resolution pixels for every variant (2 pixels in the 2×2 domain, 1 pixel in the 4×4 domain)."));

// ------------------------------------------------------------------ 4 Results
children.push(H1("4. Results"));
children.push(H2("4.1 Where the baseline loses ships"));
children.push(P(["On all 16,951 ships the hardware baseline reaches ", I("P"), SUB("d"), " = 0.891 (box), 0.846 (mask) and 0.843 (event). The strict figure is lower because the 2-pixel reference ring is contaminated: the floating-point model reaches 0.906 (box), 0.860 (mask), 0.858 (event). Table 1 and Figure 2 give the attribution of the 2,664 ships (15.72 points) that obtain no event."]));
children.push(caption("Table 1. Attribution of the ships lost by the baseline hardware prescreen (Pfa = 10⁻³, gate 0.75; 16,951 ships; event tolerance 3 px)."));
children.push(table([5200, 1300, 1400, 1460], ["Cause", "Ships", "Pd points", "Share of loss"], [
  ["D1  Own ship inside the reference ring (self-masking)", "1,586", "9.36", "59.5 %"],
  ["C   Pixels detected, but no gated event on the ship", "323", "1.91", "12.1 %"],
  ["D2  Neighbouring ship inside the ring", "257", "1.52", "9.7 %"],
  ["B   Image border: float finds it, RTL leaves outer 8 px unevaluated", "233", "1.37", "8.8 %"],
  ["E1  Low contrast, near miss (clean-sea margin −0.15…0 nat)", "135", "0.80", "5.1 %"],
  ["E2  Low contrast, deep", "69", "0.41", "2.6 %"],
  ["D3  Bright clutter inside the ring", "53", "0.31", "2.0 %"],
  ["D4  Other (variance / shape mismatch)", "8", "0.05", "0.3 %"],
  ["Total", "2,664", "15.72", "100 %"],
], { boldLast: true }));
children.push(gap());
children.push(P(["For 91 % of the lost ships the clean-sea margin is positive: against ship-free statistics the ship would have been detected. The dominant mechanism is therefore the estimator, not the visibility of the ship. Every self-masked ship is wider than the 13-pixel guard (median longest box side 46 px), and a median 27 % of the ring cells at its best pixel lie on the ship itself; the strict ",
  I("P"), SUB("d"), " does not depend on ship size class (0.82, 0.845 and 0.843 for ≤ 13, 14–26 and > 26 px), i.e. the loss is a property of bright ships that fill the ring. Truly invisible ships (below the clean-sea threshold) account for only about 1.2 points. Border ships are doubly penalised: within 8 px of the edge 788 of 2,491 ships are lost (", I("P"), SUB("d"), " = 0.68), although only the 1.37-point class B is recoverable by padding alone. We note that class B was initially mislabelled as a fixed-point effect; all 233 ships lie within 8 px of the edge and the loss stems from the RTL's valid region, not from quantisation."]));
children.push(...img("fig_p1_miss_taxonomy.png", 560, "Figure 2. Pd points lost by the baseline hardware prescreen, by cause (16,951 ships)."));

children.push(H2("4.2 Design search and the selected prescreen"));
children.push(P(["Table 2 and Figure 3 show how the strict Pd on the 842 test images (2,877 ships) changes when the attributed causes are removed one at a time. Padding the border recovers 1.5 points. Raising ", I("P"), SUB("fa"), " to 10", SUP("−2"), " buys 6.6 points at an almost unchanged load (298 events/image) because the contrast gate, not the threshold, limits the number of events; the NMS corner, however, stops landing on the ship once detection runs merge, so replacing it by a contrast-peak event gains a further 2.9 points at the cost of 85 additional events. "
  + "The decisive step is the 2×2 log-mean pooling: it halves the apparent size of the ship relative to the same window (removing most of the self-masking) and averages four speckle samples, which lowers the candidate load by a factor of four (382 → 96 events/image) while raising ", "P", SUB("d"), " to 0.991. A wider window (25/17 in the pooled domain), ", "P", SUB("fa"), " = 3×10", SUP("−2"), " and a gate of 0.6 give configuration A: ", I("P"), SUB("d"), " = 0.9986 at 190 events per image."]));
children.push(caption("Table 2. Cumulative design steps evaluated on the 842 test images (2,877 ships). Pd (event) is the strict criterion; Pd (pixel) is a detected pixel on the polygon."));
children.push(table([3500, 1500, 1450, 1450, 1460], ["Configuration", "Pd (event)", "Pd (pixel)", "Events / image", "False events / image"], [
  ["RTL baseline (17/13, valid border, Pfa 10⁻³, gate 0.75, NMS)", "0.867", "0.861", "274", "257"],
  ["+ padded border", "0.882", "0.874", "284", "267"],
  ["+ Pfa = 10⁻²", "0.948", "0.973", "298", "270"],
  ["+ contrast-peak (5×5) events", "0.977", "0.973", "383", "360"],
  ["+ 2×2 log-mean pooling (window 17/13)", "0.991", "0.997", "96", "82"],
  ["+ window 25/17", "0.993", "0.999", "101", "86"],
  ["+ Pfa = 3×10⁻², gate 0.6  (configuration A)", "0.9986", "0.9997", "190", "172"],
]));
children.push(gap());
children.push(...img("fig_design_ladder.png", 620, "Figure 3. Effect of the cumulative design steps on (a) the strict prescreen Pd and (b) the candidate events per image (test split)."));
children.push(P(["Among alternatives, full-resolution windows up to 49/33 reached 0.931 at 333 events (NMS, ", "P", SUB("fa"), " = 10", SUP("−3"), ") and 0.997 only at 434 events with peak events and ", "P", SUB("fa"), " = 3×10", SUP("−2"), "; i.e. comparable Pd at more than twice the load. Iterative censoring did not help: it lowers ",
  I("c"), SUB("1"), " and ", I("c"), SUB("2"), " around bright objects, floods the stream with detections, and the number of events per ship rises while the NMS corner disappears. 4×4 pooling yielded the lowest loads (e.g. 0.987 at 50 events per image with a 17/13 window and 3×3 peaks) but is expected to lose the smallest ships (Section 6). The Pareto frontier over all variants is shown in Figure 4."]));
children.push(...img("fig_variants_test2_pareto.png", 620, "Figure 4. Strict prescreen Pd versus candidate events per image for all 2,400 evaluated variants (842 test images). The baseline is marked by a star; the dotted line is Pd = 0.95."));

children.push(H2("4.3 Whole-data-set evaluation of the selected prescreen"));
children.push(P(["Configuration A was run on all 5,604 images. Table 3 and Figure 5 summarise the result: 16,885 of 16,951 ships receive an event within 4 px of the polygon (", I("P"), SUB("d"), " = 0.9961, 95 % CI 0.9950–0.9969); 5,553 of 5,604 images (99.09 %) have every ship delivered, and the lowest per-image value is 0.27. "
  + "Training, validation and test images behave alike (0.9954, 0.9982, 0.9972), confirming that the choice of configuration did not overfit the tuning images. Offshore scenes are essentially solved (2 misses in 8,745 ships; 0.9998), whereas inshore scenes retain 64 of the 66 misses (0.9922)."]));
children.push(caption("Table 3. Prescreen Pd (event on the ship, 4 px tolerance) of configuration A over the whole data set."));
children.push(table([2400, 1500, 1400, 1700, 2360], ["Subset", "Ships", "Missed", "Pd", "95 % Wilson CI"], [
  ["All images (5,604)", "16,951", "66", "0.9961", "0.9950 – 0.9969"],
  ["Training (3,922)", "11,855", "54", "0.9954", "0.9941 – 0.9965"],
  ["Validation (840)", "2,219", "4", "0.9982", "0.9954 – 0.9993"],
  ["Test (842)", "2,877", "8", "0.9972", "0.9945 – 0.9986"],
  ["Offshore", "8,745", "2", "0.9998", "0.9992 – 0.9999"],
  ["Inshore", "8,206", "64", "0.9922", "0.9901 – 0.9939"],
]));
children.push(gap());
children.push(P("The estimate depends on how closely an event must lie to the ship (Figure 5a, Table 4): 0.9881 if the event must fall inside the polygon, 0.9959 at 4 px and 0.9985 at 8 px. Because the discriminator inspects a window of 64×64 original pixels around each event, an event within a few pixels of the ship is sufficient for the second stage; the polygon-interior figure is reported as the most conservative reading."));
children.push(caption("Table 4. Sensitivity of the whole-data-set Pd to the event-to-polygon tolerance (recomputed from event-to-polygon distances; Table 3 uses the event labelling at the block centre and gives 0.9961 at 4 px)."));
children.push(table([2600, 1000, 1000, 1000, 1000, 1000, 1000, 760], ["Tolerance (px)", "0", "1", "2", "3", "4", "6", "8"], [
  ["Pd (16,951 ships)", "0.9881", "0.9906", "0.9922", "0.9947", "0.9959", "0.9981", "0.9985"],
], { center: true }));
children.push(gap());
children.push(...img("fig_fullsweep_overview.png", 640, "Figure 5. Configuration A over all 5,604 images. (a) Pd versus the event-to-polygon tolerance (dotted: the 4 px used elsewhere). (b) Per-image Pd histogram (log count): 99.09 % of images have all ships delivered. (c) Pd by split and scene with 95 % Wilson intervals."));

children.push(H2("4.4 Residual misses"));
children.push(P(["The 66 undelivered ships (0.39 %) are concentrated in a few conditions (Figure 6, Table 5): 64 are inshore; 31 have a contrast below 1.0 nat; 15 lie within 8 px of the image border; ships touching a neighbour (gap < 0.5 px) have ", I("P"), SUB("d"), " = 0.974, and ships smaller than 25 px (area) 0.937 (n = 64). No dependence on the local clutter level is visible."]));
children.push(caption("Table 5. Characteristics of the 66 ships missed by configuration A."));
children.push(table([4200, 1500, 3660], ["Characteristic", "Ships", "Comment"], [
  ["Inshore / offshore scene", "64 / 2", "port structures, land edges"],
  ["Polygon area ≤ 25 / 25–50 / 50–100 / 100–250 / > 250 px", "4 / 3 / 24 / 17 / 18", "no single size class dominates"],
  ["Clean contrast dx < 1.0 nat", "31", "47 % of the misses"],
  ["Within 8 px of the image border", "15", "partly outside the evaluated area"],
  ["Training / validation / test images", "54 / 4 / 8", "no sign of overfitting to the tuning set"],
], { center: false }));
children.push(gap());
children.push(...img("fig_fullsweep_by_factor.png", 640, "Figure 6. Pd of configuration A by ship and scene factor (all ships, 95 % Wilson intervals; vertical axis truncated at 0.90). The only strong effects are touching neighbours and very small ships."));
children.push(P(["The candidate load is heavy-tailed (Figure 7): the mean is 181 events per image, the median 34, the 95th percentile 1,216 and the maximum 2,622, dominated by cluttered inshore scenes. A second stage that processes one event in 0.2 ms (our quad-pixel INT8 core at 100 MHz) therefore needs up to about 0.5 s for the worst image; the event memory must be dimensioned accordingly."]));
children.push(...img("fig_event_load.png", 330, "Figure 7. Distribution of candidate events per image for configuration A (5,604 images)."));

// ------------------------------------------------------------------ 5 Discussion
children.push(H1("5. Discussion"));
children.push(P(["The attribution reframes the problem. Most missed ships are not invisible to a Weibull detector; they are lost because the reference ring of a 17×17 window is only two pixels thick, so a ship wider than the guard puts its own bright pixels into ", I("c"), SUB("1"), " and ", I("c"), SUB("2"), ", raising both the mean and the variance-dependent threshold offset. Window geometry that was tuned using the lenient box criterion cannot reveal this, because a box also ‘detects’ ships through clutter pixels at its edges or in its corners."]));
children.push(P("Pooling addresses the dominant mechanism at its root and at low cost. Averaging 2×2 blocks of log amplitude halves the ship's size in pixels relative to a window of fixed size, so the same 17- to 25-pixel window now spans a 34- to 50-pixel neighbourhood in the original geometry. It also averages four speckle samples, which narrows the clutter distribution; with the Weibull shape re-estimated from the (smaller) local variance, the threshold sits closer to the mean and a higher false-alarm rate is affordable. The latter is why the load falls by a factor of four at equal or higher Pd. A contrast-peak event keeps the candidate on the ship when detection runs merge at high Pfa, where the first-pixel NMS rule fails."));
children.push(P("Two design choices that appear natural did not help: censoring the reference ring of a first-pass detection and 4×4 pooling with wide windows. Censoring is designed to protect the estimate from isolated bright cells; here it removes ship pixels that are part of the ring consistently, lowers the estimated variance, and increases false detections without improving the event rate. This is consistent with the observation that the gain comes from a changed geometry, not from a more robust estimator."));
children.push(P(["Relation to the cascade. The second stage can only reduce the number of candidates, so the end-to-end Pd is bounded by 0.9972 on the test split with configuration A (versus 0.867 for the baseline). A discriminator retrained on the new candidate set and evaluated in floating point reaches an end-to-end Pd of 0.952 at 3.5 false events per image and 0.964 at 5.0 on the test images (preliminary; one network, no confidence intervals). Those results are reported separately and are not part of the Weibull-only evidence in this document."]));

// ------------------------------------------------------------------ 6 Limitations
children.push(H1("6. Limitations"));
children.push(bullet("The redesigned prescreen is evaluated in floating point on pooled data. In the cascade the pooled store holds the mean of 8-bit logarithmic codes, and the Weibull look-up tables and fixed-point formats must be re-derived for the pooled grid; the bit-exact hardware figure for configuration A is therefore not yet known."));
children.push(bullet("The strict criterion measures the presence of an event near the polygon (4 px), not localisation accuracy; touching ships share events. Table 4 shows the dependence on the tolerance; at zero tolerance Pd is 0.9881."));
children.push(bullet("The configuration was selected on 600 train/validation images and confirmed on one test split; the whole-data-set figure of Section 4.3 includes those images and is not an independent estimate. The test figure (0.9972, CI 0.9945–0.9986) is the independent one."));
children.push(bullet("The baseline loss attribution used a 3 px event tolerance and the redesign 4 px; like-for-like baseline numbers at 4 px are given in Table 2. The attribution relies on polygon masks whose boundaries are accurate only to about a pixel."));
children.push(bullet("HRSID mixes sensors and resolutions (0.5 m to 3 m) and does not label resolution per image; the study therefore uses pixel units and cannot state results in metres. Ships smaller than 25 px are rare (64 of 16,951) and are the ones most at risk from 4×4 pooling; the performance of that variant by ship size was not examined."));
children.push(bullet("Only the Weibull family at the existing hardware's Pfa-to-threshold mapping was studied; other clutter models and a land-masking stage, which would address the remaining inshore misses directly, were not evaluated."));

// ------------------------------------------------------------------ 7 Conclusions
children.push(H1("7. Conclusions"));
children.push(P("A per-ship, counterfactual loss attribution shows that 60 % of the detection loss of a hardware-faithful Weibull-CFAR prescreen on HRSID is self-masking by the thin reference ring, and a further 9 % by neighbouring ships; truly invisible ships account for about 1.2 points. A prescreen that detects on the 2×2 log-mean pooled image with a wider, padded reference ring, a higher false-alarm rate with a contrast gate, and contrast-peak events delivers a candidate event for 99.86 % of the ships of 842 held-out images and 99.61 % of all 16,951 ships, with fewer candidates per image than the baseline. Residual misses are inshore (0.9922 versus 0.9998 offshore), small, low-contrast or touching other ships. The next steps are bit-exact modelling of the pooled-domain prescreen, quantisation and validation on the FPGA, and a discriminator that exploits scene context to address the inshore tail."));

// ------------------------------------------------------------------ Appendix + References
children.push(H1("Appendix A. Reproducibility"));
children.push(P("All code is in the project repository. Per-ship diagnostics: _comparison/pd_study/pd_study_extract.m and analyze_misses.py. Design search: pd_variants_eval2.m and variants_report.py. Whole-data-set evaluation and figures: extract_cnn_patches_pooldet.m (configuration A: Pool 2, Mode log, Sli 25, Guard 17, Pfa 0.03, Gate 0.6, Event peak5) and report_figures.py. Result tables are under _comparison/Results/pd_study (phase-1 tables, report/) and Results/pd_variants. Written analyses: PAPER2_PD_LOSS_ANALYSIS_2026-10-03.md and the roadmap PAPER2_PD_ROADMAP_weibull-prescreen-to-near-100.md."));
children.push(H1("References"));
const refs = [
  "An, Q. et al. Ship detection in Gaofen-3 SAR images based on sea clutter distribution analysis and deep convolutional neural network. Sensors 18(2), 334 (2018). doi:10.3390/s18020334.",
  "Study on the combined application of CFAR and deep learning in ship detection. J. Indian Soc. Remote Sens. (2018). doi:10.1007/s12524-018-0787-x.",
  "Zaimbashi, A. An adaptive cell averaging-based CFAR detector for interfering targets and clutter-edge situations. Digital Signal Processing 31, 59–68 (2014). doi:10.1016/j.dsp.2014.04.005.",
  "El-Darymli, K., McGuire, P., Power, D., Moloney, C. Target detection in synthetic aperture radar imagery: a state-of-the-art survey. J. Appl. Remote Sens. 7(1), 071598 (2013). doi:10.1117/1.JRS.7.071598.",
  "An improved iterative censoring scheme for CFAR ship detection with SAR imagery. IEEE Trans. Geosci. Remote Sens. (2013). doi:10.1109/TGRS.2013.2282820.",
  "Approximate MLE based automatic bilateral censoring CFAR ship detection for complex scenes of log-normal sea clutter. Digital Signal Processing (2023). doi:10.1016/j.dsp.2023.103972.",
  "Wei, S. et al. HRSID: a high-resolution SAR images dataset for ship detection and instance segmentation. IEEE Access 8 (2020). doi:10.1109/ACCESS.2020.3005861.",
  "Superpixel-level CFAR detectors for ship detection in SAR imagery. IEEE Geosci. Remote Sens. Lett. (2018). doi:10.1109/LGRS.2018.2838263.",
  "Statistical modeling of SAR images: a survey. Sensors 10(1), 775–795 (2010). doi:10.3390/s100100775.",
  "LS-SSDD-v1.0: a deep learning dataset dedicated to small ship detection from large-scale Sentinel-1 SAR images. Remote Sens. 12(18), 2997 (2020). doi:10.3390/rs12182997.",
];
refs.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 70 }, indent: { left: 400, hanging: 400 }, children: [run(`[${i + 1}]  ${r}`, { size: 20 })] })));
children.push(P([I("Note: reference entries without an author list were located by title, venue and DOI only and need their author lists completed before submission; references [1,2] are cited for the cascade architecture, [3–6,8] for masking and censoring, [7,10] for the data sets.")], { alignment: AlignmentType.LEFT }));

const doc = new Document({
  creator: "Paper 2 working draft", title: "Raising the Detection Probability of a Weibull-CFAR Ship Prescreen",
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
