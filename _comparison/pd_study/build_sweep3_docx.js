// Builds PAPER2_Three_Pfa_report.docx : whole-data-set Weibull-only results and cascade (after-CNN) results at Pfa 1e-2, 1e-3, 1e-4.
// Numbers are read from Results/pd_sweep3/report/*.csv|json at build time.   usage: NODE_PATH=<dir with node_modules/docx> node build_sweep3_docx.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, HeadingLevel, BorderStyle, LevelFormat, Footer, PageNumber } = require("docx");

const HERE = __dirname;
const REP = path.resolve(HERE, "..", "Results", "pd_sweep3", "report");
const OUT = path.resolve(HERE, "..", "..", "PAPER2_Three_Pfa_report.docx");
const FONT = "Times New Roman";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const P = (parts, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o, children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: t, font: FONT, size: 26, bold: true })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 }, children: [new TextRun({ text: t, font: FONT, size: 23, bold: true, italics: true })] });
const B = (t) => run(t, { bold: true }); const I = (t) => run(t, { italics: true });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 }, alignment: AlignmentType.JUSTIFIED, children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
function img(file, wpx, caption) {
  const buf = fs.readFileSync(path.join(REP, file)); const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); const s = wpx / w;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true, children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(w * s), height: Math.round(h * s) }, altText: { title: file, description: caption.slice(0, 150), name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [run(caption, { size: 19 })] })];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" }; const borders = { top: border, bottom: border, left: border, right: border };
function table(cols, header, rows, size = 18) {
  const cell = (t, i, hdr) => new TableCell({ width: { size: cols[i], type: WidthType.DXA }, borders, shading: hdr ? { fill: "E8EDF5", type: ShadingType.CLEAR, color: "auto" } : undefined, margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, children: [new TextRun({ text: String(t), font: FONT, size, bold: hdr })] })] });
  return new Table({ width: { size: cols.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: cols, rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }), ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))] });
}
const caption = (t) => new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true, children: [run(t, { size: 20 })] });
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });
function csv(file) {
  const L = fs.readFileSync(path.join(REP, file), "utf8").trim().split(/\r?\n/); const hd = L[0].split(",");
  return L.slice(1).map((l) => { const v = l.split(","); const o = {}; hd.forEach((h, i) => (o[h] = v[i])); return o; });
}
const f3 = (x) => Number(x).toFixed(3), f4 = (x) => Number(x).toFixed(4), f1 = (x) => Number(x).toFixed(1), f2 = (x) => Number(x).toFixed(2);
const sci = (x) => { const v = Number(x); if (v === 0) return "0"; const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)}e${e}`; };
const W = JSON.parse(fs.readFileSync(path.join(REP, "summary_weibull_only.json"), "utf8"));
const wsel = (g, p) => W.find((r) => r.design.startsWith(g) && Math.abs(r.pfa - p) < 1e-12);
const hasCascade = fs.existsSync(path.join(REP, "cascade_operating_points.csv"));
const C = hasCascade ? csv("cascade_operating_points.csv") : [];
const cas = (pfa, tgt, subset = "test") => C.find((r) => r.pfa === pfa && Math.abs(Number(r.retention_target) - tgt) < 1e-9 && r.subset === subset);
const PF = [["1e-2", 0.01], ["1e-3", 0.001], ["1e-4", 0.0001]];
const c = [];

// ---------------------------------------------------------------------------- title + abstract
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Weibull Prescreen and Full Cascade at Three False-Alarm Settings: Whole-Data-Set Results on HRSID", font: FONT, size: 34, bold: true })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [run("Paper 2 working report, 3 October 2026 (software study). Authors: [to be completed]", { size: 20 })] }));
c.push(H1("Abstract"));
const p2 = wsel("pooled", 0.01), p3 = wsel("pooled", 0.001), p4 = wsel("pooled", 0.0001), f2_ = wsel("full", 0.01), f3_ = wsel("full", 0.001), f4_ = wsel("full", 0.0001);
let abs = `We evaluated a Weibull-CFAR ship prescreen on all 5,604 images (16,951 ships) of the HRSID data set at three nominal false-alarm probabilities, Pfa = 10⁻², 10⁻³ and 10⁻⁴, for two designs that share the same 17×17/13×13 reference ring, padded border, contrast gate (0.75) and 5×5 contrast-peak events: detection at full resolution and detection on the 2×2 log-mean pooled image. A ship counts as detected if a candidate event lies within 4 pixels of its polygon. The pooled design reaches Pd = ${f4(p2.pd_event)}, ${f4(p3.pd_event)} and ${f4(p4.pd_event)} at the three settings, against ${f4(f2_.pd_event)}, ${f4(f3_.pd_event)} and ${f4(f4_.pd_event)} at full resolution, while sending ${f1(p2.events_mean)}, ${f1(p3.events_mean)} and ${f1(p4.events_mean)} candidate events per image instead of ${f1(f2_.events_mean)}, ${f1(f3_.events_mean)} and ${f1(f4_.events_mean)}. `;
if (hasCascade) { const a2 = cas("1e-2", 0.97), a3 = cas("1e-3", 0.97), a4 = cas("1e-4", 0.97); abs += `Followed by a context-aware CNN trained for each setting, the held-out test images give a cascade Pd of ${f3(a2.cascade_pd)}, ${f3(a3.cascade_pd)} and ${f3(a4.cascade_pd)} at ${f2(a2.fa_per_img)}, ${f2(a3.fa_per_img)} and ${f2(a4.fa_per_img)} false events per image (97 % validation retention target). `; }
abs += "Per-image pixel and event false-alarm rates are reported on a logarithmic scale; the achieved pixel Pfa exceeds the nominal value and saturates near 10⁻² as the nominal Pfa is lowered. All results are floating-point software results; the pooled prescreen has not yet been validated in hardware.";
c.push(P(abs));

// ---------------------------------------------------------------------------- 1 intro/method
c.push(H1("1. Scope and method"));
c.push(P("This report extends the earlier loss-attribution and design study (PAPER2_Weibull_prescreen_Pd_report) and the explanation of the two key changes (PAPER2_Explainer_contrast_peak_and_pooling). It reports the complete sweep of the whole data set for three false-alarm settings and adds the end-to-end results after the CNN. The designs and metrics are defined below."));
c.push(H2("1.1 Prescreens"));
c.push(table([1700, 3500, 4160], ["", "Pooled design", "Full-resolution design"], [
  ["Input", "2×2 mean of the log amplitude (400×400)", "log amplitude (800×800)"],
  ["Reference ring", "17×17 minus 13×13 pooled pixels (34/26 original px)", "17×17 minus 13×13 pixels"],
  ["Border", "symmetric padding (outer pixels evaluated)", "symmetric padding"],
  ["Threshold", "T = c₁ + K(Pfa)/Ĉ (Weibull, method of log-cumulants), Pfa ∈ {10⁻², 10⁻³, 10⁻⁴}", "same"],
  ["Gate / events", "contrast x − c₁ ≥ 0.75; 5×5 contrast-peak events", "same"],
]));
c.push(gap());
c.push(H2("1.2 Measures"));
c.push(bullet([B("Pd: "), "fraction of ships with at least one candidate event within 4 full-resolution pixels of the ship polygon (touching ships share events). Wilson 95 % intervals."]));
c.push(bullet([B("Pixel Pfa: "), "detected pixels farther than 4 px from every ship, divided by the pixels farther than 4 px from every ship (per image or pooled over images)."]));
c.push(bullet([B("Event Pfa: "), "off-ship candidate events divided by the same background pixel count; “false events per image” is the unnormalised count."]));
c.push(bullet([B("Events per image: "), "total candidate events, the workload of the second stage."]));
c.push(H2("1.3 The CNN stage"));
c.push(P("For each setting a separate CNN was trained on the candidates of the 3,922 training images only. Input: the 32×32 window of the 2×2-pooled code image around the event, a 32×32 4×4-pooled context window (128×128 original pixels), and nine scalar side features (event contrast, local clutter mean and s.d., image-level log-amplitude mean and s.d., bright and dark pixel fractions, log event count). Training used a ship-level (multiple-instance) loss in which only the two highest-scoring events of each ship are positives, and periodic hard-negative mining. The accept threshold was set on the 840 validation images to retain a target fraction of the ships delivered by the prescreen, and applied unchanged to all images. Training images are in-sample for the CNN, so all cascade headline figures use the 842 held-out test images."));

// ---------------------------------------------------------------------------- 2 weibull-only
c.push(H1("2. Weibull-only results over the whole data set"));
c.push(caption("Table 1. Whole-data-set Weibull-only results (5,604 images, 16,951 ships)."));
const rows1 = [];
for (const [lab, g] of [["Pooled 2×2 log-mean", "pooled"], ["Full resolution", "full"]]) for (const [pl, pv] of PF) { const r = wsel(g, pv);
  rows1.push([`${lab}, ${pl}`, f4(r.pd_event), `${f4(r.ci_lo)}–${f4(r.ci_hi)}`, f4(r.pd_pixel), String(r.missed), `${(100 * r.images_complete).toFixed(1)} %`, f1(r.events_mean), f1(r.events_median), f1(r.off_events_mean), sci(r.pixel_pfa), sci(r.event_pfa)]); }
c.push(table([1700, 780, 1100, 780, 700, 760, 780, 760, 780, 700, 520].map((x, i) => (i === 0 ? 1560 : x)), ["Design, Pfa", "Pd", "95 % CI", "Pd (pixel)", "Missed", "Images complete", "Events / img", "Median", "Off-ship / img", "Pixel Pfa", "Event Pfa"], rows1, 15));
c.push(gap());
c.push(P(["Lowering Pfa costs the full-resolution design far more detection probability than the pooled one: from 10⁻² to 10⁻⁴ the full-resolution Pd falls by ", f1(100 * (f2_.pd_event - f4_.pd_event)), " points (", f4(f2_.pd_event), " → ", f4(f4_.pd_event), "), the pooled Pd by ", f1(100 * (p2.pd_event - p4.pd_event)), " points (", f4(p2.pd_event), " → ", f4(p4.pd_event), "). At every setting the pooled design also sends 3–4 times fewer events per image (Figure 1)."]));
c.push(...img("fig_compare_designs.png", 600, "Figure 1. Weibull-only Pd and mean candidate events per image for the full-resolution and the pooled design at the three Pfa values (whole data set)."));
c.push(H2("2.1 Per-image behaviour"));
c.push(P("Figure 2 shows, for the pooled design, the per-image Pd, the per-image pixel Pfa and the per-image event Pfa, each for all 5,604 images, ordered by the pixel Pfa of the respective setting; the black line is a rolling mean over 201 images. Most images contain every ship (Pd = 1) and the loss concentrates in the images with the highest false-alarm rates (cluttered, mostly inshore scenes). Pixel and event Pfa are plotted on a logarithmic axis; images without any off-ship event are drawn at the floor 10⁻⁸. Table 2 gives the same information as a table of decades."));
c.push(...img("fig_per_image_pooled_3pfa.png", 640, "Figure 2. Per-image Pd (left), pixel Pfa (middle, log) and event Pfa (right, log) of the pooled design at Pfa 10⁻² (top), 10⁻³ and 10⁻⁴ (bottom), 5,604 images. Images are ordered by the pixel Pfa of the respective setting."));
const D = csv("per_image_pfa_decade_table.csv");
const decRows = []; const bins = [...new Set(D.map((r) => r.per_image_range))];
for (const b of bins) { const row = [b]; for (const [pl] of PF) { const m = D.find((r) => r.design === "pooled" && r.pfa === pl && r.metric === "pixelPfa" && r.per_image_range === b); const e = D.find((r) => r.design === "pooled" && r.pfa === pl && r.metric === "eventPfa" && r.per_image_range === b); row.push(m.images, e.images); } decRows.push(row); }
c.push(caption("Table 2. Number of images (of 5,604) per decade of the per-image pixel Pfa and event Pfa, pooled design. The full table for both designs is in per_image_pfa_decade_table.csv."));
c.push(table([2400, 1160, 1160, 1160, 1160, 1160, 1160], ["Per-image range", "pixel 1e-2", "event 1e-2", "pixel 1e-3", "event 1e-3", "pixel 1e-4", "event 1e-4"], decRows));
c.push(gap());
c.push(P(["The achieved pixel Pfa is above the nominal setting: about 4×10⁻² at the nominal 10⁻², 1.7×10⁻² at 10⁻³ and 0.9×10⁻² at 10⁻⁴ (Table 1). The pooled log-mean data are not exactly Weibull and the local clutter is heterogeneous, so the detector cannot reach arbitrarily low pixel false-alarm rates by lowering the nominal Pfa; this is the practical form of the achievable-Pfa floor discussed in our earlier work on the reachability bound. What counts for the cascade is the event Pfa after the contrast gate and peak selection, which is about 10⁻⁴ per background pixel before the CNN and orders of magnitude lower after it (Section 3)."]));
c.push(...img("fig_per_image_distributions.png", 620, "Figure 3. Distributions over images of (a) the per-image pixel Pfa and (b) the per-image number of candidate events, both designs and all three settings."));
c.push(H2("2.2 Pd by split, scene, tolerance and per-image completeness"));
c.push(...img("fig_overview_3pfa.png", 600, "Figure 4. Pooled design at Pfa 10⁻² (top), 10⁻³ and 10⁻⁴ (bottom): (a) Pd versus the allowed event-to-polygon distance, (b) per-image Pd histogram (log count), (c) Pd by split and scene with 95 % Wilson intervals."));
c.push(P(`Training, validation and test images behave alike (pooled, 10⁻²: ${f4(p2.pd_train)}, ${f4(p2.pd_val)}, ${f4(p2.pd_test)}). Offshore scenes are nearly saturated (${f4(p2.pd_offshore)} at 10⁻², ${f4(p4.pd_offshore)} at 10⁻⁴), whereas inshore scenes lose most of the detection probability when Pfa is lowered (${f4(p2.pd_inshore)} at 10⁻² to ${f4(p4.pd_inshore)} at 10⁻⁴). The fraction of images in which every ship is delivered falls from ${(100 * p2.images_complete).toFixed(1)} % to ${(100 * p4.images_complete).toFixed(1)} %.`));
c.push(...img("fig_by_factor_3pfa.png", 640, "Figure 5. Pd of the pooled design by ship and scene factor at the three Pfa values (all ships, 95 % Wilson intervals; vertical axis starts at 0.7). Lower Pfa mainly loses low-contrast ships, ships touching a neighbour, and very small ships."));

// ---------------------------------------------------------------------------- 3 cascade
c.push(H1("3. After the CNN: full cascade"));
if (!hasCascade) { c.push(P("[cascade results not available at build time]")); } else {
  c.push(P("Table 3 reports the held-out test split (842 images, 2,877 ships) for several retention targets. Cascade Pd is the fraction of all ships with at least one accepted on-ship event; false events are accepted events farther than 4 px from every ship. The prescreen's own delivery rate (the ceiling of the cascade) is shown for reference."));
  c.push(caption("Table 3. Cascade results on the 842 test images for each Pfa setting and validation-retention target."));
  const rows3 = [];
  for (const [pl] of PF) for (const t of [0.90, 0.95, 0.97, 0.98, 0.99]) { const r = cas(pl, t); if (r) rows3.push([`${pl}`, `${(100 * t).toFixed(0)} %`, f3(r.delivered_pd), f3(r.cascade_pd), f2(r.fa_per_img), f1(r.accepted_per_img)]); }
  c.push(table([1400, 1500, 1900, 1700, 1500, 1360], ["Pfa", "Retention target", "Prescreen Pd (ceiling)", "Cascade Pd", "False events / img", "Accepted / img"], rows3));
  c.push(gap());
  c.push(...img("fig_weibull_vs_cascade_all.png", 640, "Figure 6. Whole data set (5,604 images, 16,951 ships), 97 % retention target: Weibull-only prescreen versus the full cascade. Dark green: all images, of which the 4,762 training images are in-sample for the CNN (optimistic); hatched: held-out images only (validation + test, 1,682 images) and the test split alone (842 images), which are the unbiased figures. (a) Pd, (b) false events per image (log)."));
c.push(...img("fig_weibull_vs_cascade.png", 560, "Figure 6b. The same comparison on the 842 test images only."));
  c.push(...img("fig_cascade_operating_curves.png", 400, "Figure 7. Cascade operating curves on the test split (retention targets 80–99 %) for the three Pfa settings."));
  c.push(...img("fig_per_image_cascade_3pfa.png", 640, "Figure 8. Per-image cascade results (97 % retention target): Pd per image, false events per image (log) and event Pfa per image (log), for the three settings. Grey: training images (in-sample for the CNN); colour: held-out images. Ordering as in Figure 2; zero false events are drawn at 0.5 and zero event Pfa at 10⁻⁸."));
  const all = (pl) => cas(pl, 0.97, "all images (train in-sample)");
  c.push(P(`Over all 5,604 images (training images are in-sample, so these are optimistic) the 97 % target gives cascade Pd of ${f3(all("1e-2").cascade_pd)}, ${f3(all("1e-3").cascade_pd)} and ${f3(all("1e-4").cascade_pd)} at ${f2(all("1e-2").fa_per_img)}, ${f2(all("1e-3").fa_per_img)} and ${f2(all("1e-4").fa_per_img)} false events per image for the three settings.`));
}

// ---------------------------------------------------------------------------- 4 discussion
c.push(H1("4. Discussion"));
c.push(P("The three settings trade detection probability against the load and false-alarm rate seen by the CNN. With pooling the load changes little with Pfa (about 70–90 events per image on average, median 11–12) because the contrast gate caps the number of events, while the prescreen Pd falls steadily as Pfa is lowered, most strongly for inshore scenes, low-contrast ships and ships adjacent to other ships. The CNN cannot recover ships the prescreen did not deliver: the cascade Pd is bounded by the delivery rate (0.990, 0.975 and 0.950 on the test images for 10⁻², 10⁻³ and 10⁻⁴), and at equal false-event rates the cascade Pd follows the same order (Table 3, Figure 7). For example at about 3 false events per image the cascade reaches 0.956 at Pfa 10⁻², 0.948 at 10⁻³ and 0.915 at 10⁻⁴ (about 2.2 false events). Lowering Pfa thus buys little workload reduction at a substantial cost in Pd; the CNN is better placed to remove false events than the prescreen to avoid them, and 10⁻² is the preferable of the three settings."));
c.push(P("These three settings use the original 17/13 pooled window and gate 0.75. The configuration reported earlier (pooled window 25/17, Pfa 3×10⁻², gate 0.6) delivers 99.7 % of the test ships and reaches a cascade Pd of 0.953 at 2.0 false events per image with the same CNN recipe, i.e. it is better than any of the three settings studied here at a comparable false-event rate; the three settings isolate the effect of Pfa and should be read as such."));
c.push(P("The full-resolution design shows why pooling was introduced: its Pd depends strongly on Pfa (0.966 to 0.748) and its event load is three to four times higher at the same setting, because the ship fills the thin reference ring and speckle is not averaged."));
c.push(H1("5. Limitations"));
c.push(bullet("Floating-point models only. The pooled-domain prescreen has not been implemented in fixed point or RTL, and the Weibull look-up tables need to be re-derived for the pooled variance range."));
c.push(bullet("The nominal Pfa is a design parameter, not the achieved false-alarm rate (Table 1); the achieved pixel Pfa saturates near 10⁻²."));
c.push(bullet("The CNN was trained once per setting (one seed); differences between settings of a few tenths of a percentage point or of the order of 0.1 false events per image are within the training noise that was not measured. Test-split numbers carry the sampling uncertainty of 2,877 ships (Wilson intervals of about ±0.5 points near 0.97)."));
c.push(bullet("Pd is an event-near-the-ship criterion with a 4-pixel tolerance, not a localisation measure; touching ships share events. The Weibull-only “false events” count includes events that would be removed by the CNN."));
c.push(H1("Appendix. Files"));
c.push(P("Whole-data-set sweep: _comparison/pd_study/pd_sweep3.m (raw per-ship/per-image output in Results/pd_sweep3/part*.mat); analysis: sweep3_report.py and sweep3_cascade.py. Tables: Results/pd_sweep3/report/summary_weibull_only.csv, per_image_weibull_only.csv (one row per image with Pd, pixel Pfa, event Pfa and log10 pixel Pfa for both designs and all settings), per_image_pfa_decade_table.csv, cascade_operating_points.csv, per_image_weibull_only_and_cascade.csv, pd_by_factor_3pfa.csv. CNN training: _comparison/cnn/train_ctx.py (runs pcC2_s1, pcC3_s1, pcC4_s1)."));

const doc = new Document({
  creator: "Paper 2 working report", title: "Weibull prescreen and cascade at three false-alarm settings",
  styles: { default: { document: { run: { font: FONT, size: 22 } } }, paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } } ] },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, right: 1300, bottom: 1300, left: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 18 })] })] }) }, children: c }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
