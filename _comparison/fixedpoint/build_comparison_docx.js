// Builds PAPER2_HRSID_literature_comparison.docx (software model; FPGA columns marked pending).
// usage: NODE_PATH=<dir with node_modules/docx> node build_comparison_docx.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, HeadingLevel, BorderStyle, LevelFormat, Footer, PageNumber, PageOrientation, ImageRun } = require("docx");

const RES = path.resolve(__dirname, "..", "Results", "fixedpoint");
const OUT = path.resolve(__dirname, "..", "..", "PAPER2_HRSID_literature_comparison.docx");
const FONT = "Times New Roman";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 21, ...o });
const P = (parts, o = {}) => new Paragraph({ spacing: { after: 110, line: 270 }, alignment: AlignmentType.JUSTIFIED, ...o, children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 260, after: 110 }, children: [new TextRun({ text: t, font: FONT, size: 26, bold: true })] });
const B = (t) => run(t, { bold: true }); const I = (t) => run(t, { italics: true });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 50, line: 260 }, alignment: AlignmentType.JUSTIFIED, children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)) });
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" }; const borders = { top: border, bottom: border, left: border, right: border };
function table(cols, header, rows, size = 16, hl = []) {
  const cell = (t, i, hdr, shade) => new TableCell({ width: { size: cols[i], type: WidthType.DXA }, borders, shading: hdr ? { fill: "E8EDF5", type: ShadingType.CLEAR, color: "auto" } : shade ? { fill: "FFF4D6", type: ShadingType.CLEAR, color: "auto" } : undefined, margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ alignment: i <= 1 && !hdr ? AlignmentType.LEFT : AlignmentType.CENTER, children: [new TextRun({ text: String(t), font: FONT, size, bold: hdr })] })] });
  return new Table({ width: { size: cols.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: cols,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }), ...rows.map((r, k) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, hl.includes(k))) }))] });
}
const caption = (t) => new Paragraph({ spacing: { before: 150, after: 70 }, keepNext: true, children: [run(t, { size: 20 })] });
const note = (t) => new Paragraph({ spacing: { before: 40, after: 120 }, alignment: AlignmentType.JUSTIFIED, children: [run(t, { size: 17, italics: true })] });
const csv = (f) => { const L = fs.readFileSync(path.join(RES, f), "utf8").trim().split(/\r?\n/); const h = L[0].split(","); return L.slice(1).map((l) => { const v = l.split(","); const o = {}; h.forEach((k, i) => (o[k] = v[i])); return o; }); };
const f3 = (x) => Number(x).toFixed(3), f2 = (x) => Number(x).toFixed(2), f1 = (x) => Number(x).toFixed(1), pc = (x) => (100 * Number(x)).toFixed(1);

const D = JSON.parse(fs.readFileSync(path.join(RES, "comparison_data.json"), "utf8"));
const Q8 = csv("software_metrics_pf_plain_q8.csv"), FL = csv("software_metrics_pf_full_s1.csv"), CX = csv("software_metrics_pf_full_q8.csv");
const cx = (t) => CX.find((r) => r.target !== "-" && Math.abs(Number(r.target) - t) < 1e-9);
const q8 = (t) => Q8.find((r) => r.target !== "-" && Math.abs(Number(r.target) - t) < 1e-9), fl = (t) => FL.find((r) => r.target !== "-" && Math.abs(Number(r.target) - t) < 1e-9);
const A_fx = D["A_25_17_3e-2|fixed"], A_fl = D["A_25_17_3e-2|float_exact"];
const pre = Q8[0];
const FS = csv("full_ctx_metrics.csv"), SS = csv("seeds_summary.csv"), SPR = csv("seeds_paired.csv");
const ci = (v, lo, hi, g) => g(v) + " (" + g(lo) + "-" + g(hi) + ")";
const fsr = (sub) => FS.filter((r) => r.subset === sub);

// ---------------------------------------------------------------- Table 1: literature, software accuracy on HRSID
const T1 = [
  ["[1]", "Faster R-CNN, ResNet-50-FPN (HRSID baseline, 2020, IEEE Access)", "two-stage CNN", "AP 63.5, AP50 86.7, AP75 73.3", "330.2 MB", "0.074 s/img, RTX 2080"],
  ["[1]", "Cascade R-CNN, ResNet-101-FPN (baseline)", "multi-stage CNN", "AP 66.8, AP50 87.9, AP75 76.6", "704.8 MB", "0.109 s/img, RTX 2080"],
  ["[1]", "Hybrid Task Cascade, ResNet-101-FPN (baseline)", "multi-stage CNN", "AP 68.4, AP50 87.7, AP75 78.8", "791.6 MB", "0.156 s/img, RTX 2080"],
  ["[1]", "HRSDNet, HRFPN-W40 (best baseline)", "multi-stage CNN", "AP 69.4, AP50 89.3, AP75 79.8", "728.2 MB", "0.154 s/img, RTX 2080"],
  ["[3]", "Slimming SAR ship detector (2020, JSTARS)", "pruned + distilled CNN", "n.r. in abstract", "2.8 MB model", ">200 FPS"],
  ["[4]", "LMSD-YOLO (2022, Remote Sens.)", "YOLO, lightweight", "n.r. in abstract (SSDD, HRSID, GFSDD)", "7.6 MB", "68.3 FPS, Jetson AGX Xavier"],
  ["[5]", "MEA-Net (2022, Remote Sens.)", "lightweight CNN", "AP +3.64 pts over baseline", "0.96 M params, 2.80 GFLOPs", "6.31 FPS, Jetson Nano"],
  ["[6]", "LRTransDet (2023, Remote Sens.)", "lightweight ViT + CNN", "93.9 (\"detection accuracy\")", "3.07 M params", "75.8 FPS"],
  ["[7]", "GL-DETR (2024, GRSL)", "transformer (DETR)", "+4.89 pts over baseline (absolute n.r.)", "n.r.", "n.r."],
  ["[8]", "DBW-YOLO (2024, JSTARS)", "YOLOv7-tiny based", "mAP 88.84", "n.r.", "n.r."],
  ["[9]", "ELLK-Net (2024, TGRS)", "anchor-free, large kernel", "AP50 90.6 (horizontal box), 79.7 (rotated box)", "n.r.", "+48.7 % FPS after re-parameterisation, Jetson NX"],
  ["[10]", "LH-YOLO (2024, Remote Sens.)", "YOLOv8n based", "mAP50 96.6 (+1.4 over YOLOv8n)", "1.862 M params, -23.8 % FLOPs vs YOLOv8n", "n.r."],
  ["[11]", "AC-YOLO (2025, PLoS ONE)", "YOLO11n based", "AP50 89.3, AP 67.4 (YOLO11n: 89.0 / 65.9)", "1.8 M params, 5.4 GFLOPs, 3.75 MB", "n.r."],
  ["[12]", "RLE-YOLO (2025, IEEE Access)", "YOLOv8s based", "mAP50 98.4 / 93.9 (HRSID / SSDD as ordered in the abstract; see note)", "-43.9 % params, -34.5 % FLOPs vs YOLOv8s", "n.r."],
  ["[13]", "PPDM-YOLO (2025, JSTARS)", "YOLO11n based", "mAP50 93.7, mAP50-95 70.3", "-34.7 % params vs YOLO11n", "n.r."],
  ["[14]", "SMEP-DETR (2025, Remote Sens.)", "transformer (DETR)", "mAP 93.2", "n.r.", "n.r."],
  ["[15]", "Enhanced YOLO for small ships (2025, Remote Sens.)", "YOLOv8 based", "AP50 91 (small-ship AP about +2)", "n.r.", "n.r."],
  ["[16]", "LSD-Det (2025, IEEE Access)", "YOLOv8n based", "+0.8 mAP50, +1.3 mAP50-95 over YOLOv8n", "-65.7 % params, -20.7 % GFLOPs", "faster than baseline (n.r.)"],
  ["[17]", "MCEM (2025, Sensors)", "anchor-free", "AP_S (small ships) 45.1 (+2.3 over YOLOv8)", "n.r.", "\"real-time\" (n.r.)"],
  ["[18]", "HF-Head (2026, IEEE Access)", "YOLO + lightweight head", "mAP50-95 0.754", "-13.2 % params, -28.0 % GFLOPs", "FPGA RTL, P3 head: 3.81x lower latency, 240.66 mJ/frame"],
  ["[21]", "Adaptive CFAR, intensity-texture attention + generalised-gamma (2022, Sensors, PMC9659258)", "CFAR (model based)", "5 HRSID / LS-SSDD scenes, figure of merit 1.0 in 4 of 5 (CA-CFAR 0.56-0.8); no dataset-level Pd / Pfa", "-", "-"],
];
// ---------------------------------------------------------------- Table 2: this work, software
const row = (sys, op, r) => [sys, op, pc(r.recall), pc(r.recall_inshore), pc(r.recall_offshore), f2(r.FA_per_img), f2(r.events_per_img), f3(r.precision), f3(r.F1)];
const T2 = [
  ["Weibull prescreen only (float model)", "whole data set, all 5,604 images", pc(A_fl.recall), pc(A_fl.inshore), pc(A_fl.offshore), f1(A_fl.offship_per_img), f1(A_fl.events_per_img), "-", "-"],
  ["Weibull prescreen only (integer = RTL model)", "whole data set, all 5,604 images", pc(A_fx.recall), pc(A_fx.inshore), pc(A_fx.offshore), f1(A_fx.offship_per_img), f1(A_fx.events_per_img), "-", "-"],
  row("Weibull prescreen only (integer)", "test split (842 images)", pre),
  row("Cascade: prescreen + INT8 CNN", "val-calibrated, 90 % ship retention", q8(0.9)),
  row("Cascade: prescreen + INT8 CNN", "val-calibrated, 95 % ship retention", q8(0.95)),
  row("Cascade: prescreen + INT8 CNN", "val-calibrated, 97 % ship retention", q8(0.97)),
  row("Cascade: prescreen + INT8 CNN", "val-calibrated, 98 % ship retention", q8(0.98)),
  row("Cascade: prescreen + INT8 context CNN (hardware)", "val-calibrated, 90 % ship retention", cx(0.9)),
  row("Cascade: prescreen + INT8 context CNN (hardware)", "val-calibrated, 95 % ship retention", cx(0.95)),
  row("Cascade: prescreen + INT8 context CNN (hardware)", "val-calibrated, 97 % ship retention", cx(0.97)),
  row("Cascade: prescreen + INT8 context CNN (hardware)", "val-calibrated, 98 % ship retention", cx(0.98)),
  row("Cascade, float, context tower + side features (software ceiling)", "val-calibrated, 95 % ship retention", fl(0.95)),
  row("Cascade, float, context tower + side features (software ceiling)", "val-calibrated, 97 % ship retention", fl(0.97)),
];
// ---------------------------------------------------------------- Table 4: hardware
const T4 = [
  ["[2]", "Ship detection for on-board processing (classic: sea-land, top-hat, context, SIFT + SVM)", "Xilinx XQR5VFX130 (space grade)", "4096 x 4096, 8 bit", "95.12 % (own 3,218-image set; equals the MATLAB version)", "0.21 s", "73,061 LUT, 56,600 reg, 727 LUTRAM", "92", "31", "100 MHz; chip variant 1.32 W @ 50 MHz"],
  ["[2] ref. [10]", "CFAR ship detection, parallel-pipelined", "Zynq UltraScale", "n.r.", "n.r.", "16 s", "41 K LUT", "145", "260", "n.r."],
  ["[2] ref. [11]", "CNN ship detection, fully pipelined", "Virtex-7 XC7VX690T", "416 x 416", "94 %", "n.r.", "123 K LUT", "263", "1009", "n.r."],
  ["[2] ref. [12]", "Improved YOLOv2", "Zynq XC7Z035", "1024 x 1024", "n.r.", "3.4 s", "83 K LUT", "369", "192", "n.r."],
  ["[19]", "HE-BiDet binary-NN detector (SSDD 92.7 % mAP; SAR-Ship 90.12 % mAP)", "Xilinx XC7VX690T", "SAR-Ship / SSDD chips", "mAP 92.7 / 90.12 % (not HRSID)", "80.5 / 189.3 FPS", "n.r. here", "n.r. here", "15.8x fewer DSP than prior SOTA", "n.r."],
  ["[20]", "Customised YOLOv8 for on-satellite vessel detection", "Kria KV260 class FPGA", "xView3-SAR scenes", "detection F1 within 2.1 % of the best (not HRSID)", "40,000 km2 scene < 1 min", "n.r.", "n.r.", "n.r.", "average 7.0 W"],
  ["[18]", "HF-Head detection head (HRSID mAP50-95 0.754)", "FPGA (RTL of the P3 head only)", "head level", "see Table 1", "3.81x latency speed-up", "n.r.", "n.r.", "near-iso-DSP", "240.66 mJ/frame"],
  ["This work: prescreen only (Quartus fit incl. 160 kB store)", "Pooled Weibull prescreen, integer, bit-exact RTL", "Cyclone V 5CSXFC6D6F31C6", "800 x 800, 8 bit", "see Table 2 (recall 99.6 % of all ships)", "9.01 ms (measured on board, constant)", "826 ALM (2 %), 1,345 reg", "181 M10K (125 = store)", "8", "100 MHz, +0.98 ns slack; power n.m."],
  ["This work: full cascade, context CNN (prescreen + INT8 fine/context/side network), JTAG-fed", "Pooled Weibull prescreen + INT8 context network", "Cyclone V DE10-Standard", "800 x 800, 8 bit", "software model, bit-exact on 1,000 board frames (71,380 events): 96.2 % recall @ 2.64 FA/img (97 % target) or 93.6 % @ 1.28 (95 % target), test split", "after last pixel: mean 32.7 ms, median 16.6 ms, p95 94.6 ms (1,000 images, measured)", "11,856 ALM (28 %), 18,632 reg", "394 M10K (71 %)", "96", "100 MHz CNN / 50 MHz stream, slack +0.78 / +3.97 ns; power n.m."],
  ["This work: full cascade, single-tower CNN, JTAG-fed", "Pooled Weibull prescreen + single-tower INT8 CNN", "Cyclone V DE10-Standard", "800 x 800, 8 bit", "software model, bit-exact on 1,000 board frames (71,380 events): 95.4 % recall @ 3.24 FA/img (test split)", "after last pixel: mean 23.8 ms, median 13.8 ms, p95 62.4 ms (1,000 images, measured)", "11,172 ALM (27 %), 17,389 reg", "397 M10K (72 %)", "74", "100 MHz CNN / 50 MHz stream, slack +1.49 / +3.98 ns; power n.m."],
];
const T5 = [
  ["ALMs", "27,249 (65 %)", "11,172 (27 %)", "11,856 (28 %)", "57 %"],
  ["Registers", "37,823", "17,389", "18,632", "51 %"],
  ["M10K blocks", "487 (88 %)", "397 (72 %)", "394 (71 %)", "19 %"],
  ["DSP blocks", "79 (71 %)", "74 (66 %)", "96 (86 %)", "-22 % (more DSPs)"],
  ["Weibull stage alone: ALMs / M10K / DSP", "17,411 / 118 / 13 (streaming core, 800 px lines, 17/13 window)", "826 / 181 / 8 (incl. 160 kB frame store)", "same prescreen", "95 % fewer ALMs, 38 % fewer DSP; +63 M10K (the store, shared with the CNN)"],
  ["Candidate events per image (first 1,000 images)", "mean 277, median 87", "mean 71, median 23", "mean 71, median 23", "74 % fewer (config A peak events)"],
  ["Time after last pixel, mean / median / p95 (measured)", "57.4 / 18.0 / 357 ms", "23.8 / 13.8 / 62 ms", "32.7 / 16.6 / 95 ms", "43 % / 8 % / 74 % shorter"],
  ["CNN cycles per candidate (RTL simulation)", "19,658 (DEEP)", "19,658", "28,003 (+4,096 context-patch reads)", "-"],
  ["Ship retention at about 2 false events per image (hardware-exact, test split)", "87.9 % @ 1.98 (DEEP INT8)", "93.4 % @ 2.07", "94.8 % @ 1.81", "+6.9 points"],
  ["Border handling", "outer 8 px unevaluated", "mirrored padding, full frame", "same", "-"],
  ["Pixel stream", "gap-free (clock gating needed for a slow host)", "gaps tolerated", "gaps tolerated", "-"],
];
const REFS = [
  "[1] S. Wei, X. Zeng, Q. Qu, M. Wang, H. Su, J. Shi, \"HRSID: A high-resolution SAR images dataset for ship detection and instance segmentation,\" IEEE Access, vol. 8, pp. 120234-120254, 2020, doi:10.1109/ACCESS.2020.3005861.",
  "[2] M. Xu, L. Chen, H. Shi, Z. Yang, J. Li, T. Long, \"FPGA-based implementation of ship detection for satellite on-board processing,\" IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 15, pp. 9733-9745, 2022, doi:10.1109/JSTARS.2022.3218440 (REF2 in the project library).",
  "[3] S. Chen, R. Zhan, W. Wang, J. Zhang, \"Learning slimming SAR ship object detector through network pruning and knowledge distillation,\" IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., 2020, doi:10.1109/JSTARS.2020.3041783.",
  "[4] Y. Guo, S. Chen, R. Zhan, W. Wang, J. Zhang, \"LMSD-YOLO: A lightweight YOLO algorithm for multi-scale SAR ship detection,\" Remote Sens., vol. 14, no. 19, 4801, 2022, doi:10.3390/rs14194801.",
  "[5] Y. Guo, L. Zhou, \"MEA-Net: A lightweight SAR ship detection model for imbalanced datasets,\" Remote Sens., vol. 14, no. 18, 4438, 2022, doi:10.3390/rs14184438.",
  "[6] K. Feng, L. Li, X. Wang et al., \"LRTransDet: A real-time SAR ship-detection network with lightweight ViT and multi-scale feature fusion,\" Remote Sens., vol. 15, no. 22, 5309, 2023, doi:10.3390/rs15225309.",
  "[7] C. Li, Y. Hei, L. Xi et al., \"GL-DETR: Global-to-local transformers for small ship detection in SAR images,\" IEEE Geosci. Remote Sens. Lett., 2024, doi:10.1109/LGRS.2024.3461212.",
  "[8] X. Tang, J. Zhang, Y. Xia et al., \"DBW-YOLO: A high-precision SAR ship detection method for complex environments,\" IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., 2024, doi:10.1109/JSTARS.2024.3376558.",
  "[9] J. Shen, L. Bai, Y. Zhang et al., \"ELLK-Net: An efficient lightweight large kernel network for SAR ship detection,\" IEEE Trans. Geosci. Remote Sens., 2024, doi:10.1109/TGRS.2024.3451399.",
  "[10] Q. Cao, H. Chen, S. Wang et al., \"LH-YOLO: A lightweight and high-precision SAR ship detection model based on the improved YOLOv8n,\" Remote Sens., vol. 16, no. 22, 4340, 2024, doi:10.3390/rs16224340.",
  "[11] R. He, D. Han, X. Shen et al., \"AC-YOLO: A lightweight ship detection model for SAR images based on YOLO11,\" PLoS ONE, 2025, doi:10.1371/journal.pone.0327362 (full text in the project vault).",
  "[12] Y. Xu, X. Xue, C. Li et al., \"RLE-YOLO: A lightweight and multiscale SAR ship detection based on improved YOLOv8,\" IEEE Access, 2025, doi:10.1109/ACCESS.2025.3550541.",
  "[13] H.-J. He, T. Hu, S. Xu et al., \"PPDM-YOLO: A lightweight algorithm for SAR ship image target detection in complex environments,\" IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., 2025, doi:10.1109/JSTARS.2025.3602497.",
  "[14] C. Yu, Y. Shin, \"SMEP-DETR: Transformer-based ship detection for SAR imagery with multi-edge enhancement and parallel dilated convolutions,\" Remote Sens., vol. 17, no. 6, 953, 2025, doi:10.3390/rs17060953.",
  "[15] T. Guan, S. Chang, C. Wang et al., \"SAR small ship detection based on enhanced YOLO network,\" Remote Sens., vol. 17, no. 5, 839, 2025, doi:10.3390/rs17050839.",
  "[16] Z. Wang, B. Qin, S. Gao, \"LSD-Det: A lightweight detector for small ship targets in SAR images,\" IEEE Access, 2025, doi:10.1109/ACCESS.2025.3593021.",
  "[17] H. Chen, M. He, Z. Yang et al., \"MCEM: Multi-cue fusion with clutter invariant learning for real-time SAR ship detection,\" Sensors, vol. 25, no. 18, 5736, 2025, doi:10.3390/s25185736.",
  "[18] Z. Chen, H. Hu, W. Cheng et al., \"HF-Head: A lightweight head design for SAR small-ship detection with algorithm-hardware co-optimization,\" IEEE Access, 2026, doi:10.1109/ACCESS.2026.3699472.",
  "[19] D. Zhang, Z. Liang, R. Cen et al., \"HE-BiDet: A hardware efficient binary neural network accelerator for object detection in SAR images,\" Micromachines, vol. 16, no. 5, 549, 2025, doi:10.3390/mi16050549 (project vault).",
  "[20] C. Laganier, L. Fletcher, E. Kwan et al., \"Efficient SAR vessel detection for FPGA-based on-satellite sensing,\" 2025, doi:10.1145/3769102.3772713 (project vault).",
  "[21] \"Adaptive CFAR method for SAR ship detection using intensity and texture feature fusion attention contrast mechanism,\" Sensors, 2022, PMC9659258 (project vault; author list not recorded in the vault note).",
];

const kids = [];
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Comparison of the Weibull-Prescreen / CNN Cascade with Recent HRSID Ship-Detection Literature", font: FONT, size: 32, bold: true })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run("Software model (float, integer prescreen and INT8 CNN) and FPGA implementation (DE10-Standard). Paper 2 working document, 2026-10-04.", { italics: true })] }));

kids.push(H1("1. Purpose and status"));
kids.push(P("This document sets the proposed cascade (a pooled-domain Weibull CFAR prescreen followed by a small INT8 CNN) next to recently published SAR ship detectors that report results on HRSID [1], the data set used throughout this work (5,604 images, 16,951 ships). The literature set is not restricted to CFAR: it covers YOLO-family, transformer and classical two-stage detectors, model-based CFAR detectors, and the few FPGA or edge implementations that name HRSID. Accuracy is compared on the software model; the hardware is bit-exact to it (Section 7). Hardware columns (resources, clock, latency; power not yet measured) follow the layout of the reference FPGA paper [2] and are filled from the Quartus fits and the on-board sweep."));
kids.push(P([B("Provenance of the numbers. "), "Literature values are taken from the abstract or full text of each cited paper as retrieved on 2026-10-03 (HRSID baseline table read from the PDF in the project folder; AC-YOLO from the full text in the vault; the others from the records returned by the scholarly-search APIs, i.e. abstracts). \"n.r.\" means the figure is not reported in the text consulted, not that it does not exist in the paper. No value has been estimated or back-filled. Our own values come from the scripts listed at the end of this document."]));

kids.push(H1("2. How the reference FPGA paper [2] performs its comparisons"));
kids.push(P("Xu et al. [2] port a classical ship-detection chain (sea-land segmentation, top-hat attention, local-context facilitation, SIFT + SVM) to a space-grade FPGA for 4096 x 4096 images. Their evaluation is almost entirely a hardware-efficiency comparison:"));
kids.push(bullet([B("Same algorithm, different implementation (Table VII). "), "Slice, register, LUT, LUTRAM, BRAM and DSP usage of the proposed design against a conventional pipelined FPGA design and an HLS design of the identical algorithm on a Virtex-7 XC7VX690T, with relative reduction ratios (e.g. 63 % fewer LUTs and 80 % fewer DSPs than the pipeline design)."]));
kids.push(bullet([B("Processing time on other platforms (Fig. 13). "), "The same algorithm in C on a Xeon E5-2650 workstation and on a TS201 DSP (800 MHz): the FPGA is more than 4x faster than the CPU and more than 8x faster than the DSP."]));
kids.push(bullet([B("Previous FPGA ship detectors (Table VIII). "), "A single table of platform, image size, detection rate, time, LUTs, DSPs and BRAMs for [10], [11], [12] and the proposed work, reproduced in our Table 4."]));
kids.push(bullet([B("Real-time ratio. "), "A system-level indicator, time ratio = T_transmission / max(T_imaging, T_detection) = 1.09 > 1, plus a fabricated-chip data sheet (1.32 W at 50 MHz, < 1 s per 4k x 4k image)."]));
kids.push(bullet([B("Accuracy. "), "A single detection rate (95.12 % on a 3,218-image set built from GF-3 A, SSDD and SDD-SAR data), reported to show that the hardware version equals the MATLAB version. No precision, false-alarm count, AP or comparison with other detectors on a public benchmark is given."]));
kids.push(P("Two consequences for this work. First, the accuracy side of such a paper is deliberately thin, so the accuracy comparison below has to come from the detector literature, not from [2]. Second, the hardware side of our paper should follow [2]: same-algorithm baselines (for us: the earlier streaming Weibull core, the prescreen alone, and the full cascade), time per frame on CPU / GPU / FPGA, a resource table in the layout of [2]'s Table VIII, and a real-time indicator. Those entries are reserved in Table 4."));

kids.push(H1("3. Evaluation protocols and why the numbers are not interchangeable"));
kids.push(table([2300, 6200, 6500], ["Aspect", "HRSID detector literature (Tables 1)", "This work (Table 2)"], [
  ["Output", "Bounding box (some papers: rotated box) with confidence", "Point event on the 2x2-pooled grid (800 x 800 image -> 400 x 400 grid), accept / reject per event"],
  ["Hit criterion", "IoU >= 0.5 (AP50) or averaged IoU 0.5-0.95 (AP, mAP50-95); the HRSID baselines were filtered at IoU 0.7", "an accepted event within 4 full-resolution pixels of the ship polygon (lenient: no box size or shape is predicted)"],
  ["Headline metric", "AP50 / mAP: area under the precision-recall curve over all confidence thresholds", "ship recall at a chosen operating point, false events per image, ship-level precision and F1 at that point"],
  ["Data split", "HRSID official 65 % / 35 % split in [1]; many papers use their own (e.g. 7:2:1 in [11])", "our split by image (70 / 15 / 15 %, fixed seed); thresholds from validation images, numbers on the 842 test images (2,877 ships); the prescreen alone also on all 5,604 images"],
  ["Training data", "trained and tested on HRSID boxes", "prescreen has no training; CNN trained on candidate events of the training images only"],
  ["Input scale", "resized (e.g. 640 or 1000 px) before inference", "native 800 x 800, pooled in the datapath"],
], 17));
kids.push(note("Consequence: AP50 is an area-under-curve number and our recall / FA pair is one point on a curve, so neither can be converted into the other. The closest literal equivalent of AP50 in this work would be a full precision-recall curve of ship-level precision against recall, obtainable by sweeping the CNN threshold; it is not computed here because our events are points, not boxes, and an IoU-based AP would require adding box regression."));

kids.push(H1("4. Literature results on HRSID (software)"));
kids.push(caption("Table 1. Recent and baseline results on HRSID, exactly as reported by each source."));
kids.push(table([620, 3900, 1800, 3500, 2500, 2700], ["Ref.", "Method (year, venue)", "Family", "HRSID accuracy as reported", "Model cost as reported", "Speed / platform as reported"], T1, 15));
kids.push(note("[1] baselines evaluated with an IoU threshold of 0.7 for filtering; in [1] the same detectors reach inshore AP50 of 78.3-82.7 and offshore AP50 of 98.0-99.0 (Table 5 of [1]; inshore scenes are 18.4 % of the test set). [12]: the abstract lists \"98.4 % and 93.9 % mAP50\" for \"the HRSID and SSDD datasets\"; 98.4 % would be far above every other HRSID result in this table, so the order may be reversed in the abstract and the value should be checked in the full paper before it is quoted. [4], [5], [6] state accuracy gains or model cost in the abstract only; \"detection accuracy\" in [6] is taken to be mAP50 but the abstract does not say. [21] is not a data-set-level evaluation."));

kids.push(H1("5. This work: software results"));
kids.push(P(["Recall is the fraction of ships with at least one accepted event within 4 px of the ship polygon; ships never delivered by the prescreen count as missed. A false event is an accepted event farther than 4 px from every ship; ", I("precision"), " = detected ships / (detected ships + false events); F1 is computed from these. Inshore / offshore labels are those of HRSID scenes. The INT8 CNN is the single-tower network (65,633 parameters) trained with the ship-level loss and hard-negative mining on hardware-exact candidates and quantised with per-channel int8 weights; its integer scores are bit-exact with the RTL datapath (golden regression of the CNN core: 64 of 64 vectors, 0 mismatches)."]));
kids.push(caption("Table 2. Detection performance of the proposed system (config A prescreen: pooled 2x2, 25/17 ring window, Pfa 3e-2, gate 0.6, 5x5 contrast-peak events)."));
kids.push(table([3600, 2900, 950, 950, 950, 1000, 1100, 900, 700], ["System", "Subset / operating point", "Recall %", "Inshore recall %", "Offshore recall %", "False events / img", "Accepted events / img", "Precision", "F1"], T2, 15, [4]));
kids.push(note("Whole-data-set rows: 16,951 ships (8,206 inshore, 8,745 offshore); for the prescreen alone every event is accepted, so \"false events\" are the off-ship events (precision and F1 are not meaningful there). Test-split rows: 842 images, 2,877 ships (1,466 inshore, 1,411 offshore). The prescreen reaches 99.69 % of the test ships (8-9 never delivered). The shaded rows are the 95 % operating points of the single-tower and the context network. The last two rows are the float network of the first context study (side features as floating-point numbers), kept as a software reference; the hardware context network uses integer side-feature codes."));
kids.push(P([B("Reading Table 2 against Table 1. "), "(i) The prescreen alone delivers ", pc(A_fx.recall), " % of all HRSID ships (inshore ", pc(A_fx.inshore), " %, offshore ", pc(A_fx.offshore), " %) at ", f1(A_fx.events_per_img), " candidate events per image; in [1] the best detector's offshore AP50 is 99.0 % and inshore 82.7 %, so the model-based stage alone is at least competitive in recall, but it is not a detector: it cannot be used without the CNN because of 166 off-ship events per image. (ii) After the INT8 CNN the system keeps ", pc(q8(0.95).recall), " % of ships (inshore ", pc(q8(0.95).recall_inshore), " %, offshore ", pc(q8(0.95).recall_offshore), " %) at ", f2(q8(0.95).FA_per_img), " false events per image, ship-level precision ", f3(q8(0.95).precision), ". Published box detectors with AP50 near 90 % typically operate at much higher precision at similar recall (the PR curves in Fig. 8-9 of [1] stay near 1.0 precision up to roughly 0.6 recall); our precision at 95 % recall is clearly lower, and ", I("we do not claim to beat these detectors on accuracy"), ". (iii) Inshore remains the weak tail in both worlds: in [1] inshore AP50 is 20 points below offshore; here the cascade's inshore recall is ", pc(q8(0.97).recall_inshore), " % against ", pc(q8(0.97).recall_offshore), " % offshore at the 97 % target. (iv) The context-tower network is now in hardware: at the 97 % target it keeps ", pc(cx(0.97).recall), " % of ships at ", f2(cx(0.97).FA_per_img), " false events per image (single tower: ", pc(q8(0.97).recall), " % at ", f2(q8(0.97).FA_per_img), "), and at the 95 % target ", pc(cx(0.95).recall), " % at ", f2(cx(0.95).FA_per_img), " (single tower ", pc(q8(0.95).recall), " % at ", f2(q8(0.95).FA_per_img), "); at about 1.3 false events per image the gain is about 2 points of recall, and the inshore recall improves from ", pc(q8(0.97).recall_inshore), " % to ", pc(cx(0.97).recall_inshore), " % at the 97 % target (unchanged at the 95 % target: ", pc(cx(0.95).recall_inshore), " %). This is the original run; the paired comparison over five runs in Table 2c shows the robust effect is fewer false events at about equal recall, not the two points of recall."]));

kids.push(H1("5a. Whole data set on the board: Weibull only versus Weibull + CNN"));
kids.push(P(["The context cascade was run on the DE10-Standard over all 5,604 HRSID images (1,015,171 events, every frame bit-exact against the golden models, prescreen 9.01 ms per frame). One sweep gives both systems: ", B("Weibull only"), " accepts every prescreen event; ", B("Weibull + CNN"), " accepts events whose integer logit reaches the threshold, which is applied to the stored logits at the four validation-calibrated values (90 / 95 / 97 / 98 % ship retention on the validation images). The CNN was trained on the train images, so only the test rows are held-out; train and validation rows are in-sample for the CNN (the prescreen has no training). 95 % confidence intervals come from 1,000 bootstrap resamples of images."]));
kids.push(caption("Table 2a. Whole-data-set evaluation of the on-board sweep (context cascade, config A prescreen), by subset."));
const SUBN = { all: "all 5,604", train: "train 3,922", val: "val 840", test: "test 842" };
const sysName = (s) => s.replace("Weibull only (prescreen)", "Weibull only").replace("cascade @ ", "Weibull + CNN @ ").replace(" val retention (theta ", " (theta ");
const T6 = [], hl6 = [];
["all", "train", "val", "test"].forEach((sub) => fsr(sub).forEach((r) => { if (r.system.includes("97 %")) hl6.push(T6.length); T6.push([SUBN[sub], sysName(r.system), ci(100 * r.recall, 100 * r.recall_lo, 100 * r.recall_hi, f1), pc(r.recall_inshore), pc(r.recall_offshore), ci(r.FA_per_img, r.FA_per_img_lo, r.FA_per_img_hi, f2), f3(r.precision), f3(r.F1)]); }));
kids.push(table([1300, 3700, 2500, 1150, 1150, 2600, 1000, 800], ["Subset (images)", "System", "Recall % (95 % CI)", "Inshore %", "Offshore %", "False events / img (95 % CI)", "Precision", "F1"], T6, 15, hl6));
kids.push(note("Recall = ships with at least one accepted event within 4 px of the ship polygon (16,951 ships on all images; 2,877 on the test split). Weibull-only rows: every event accepted, so the false events are all off-ship events and precision is not meaningful as a detector figure. Shaded rows: the 97 % operating point. The test rows reproduce the software evaluation of the INT8 network (96.2 % at 2.64 false events per image at the 97 % target)."));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(path.resolve(__dirname, "..", "Figures", "fixedpoint_full_sweep.png")), transformation: { width: 900, height: 360 }, altText: { title: "Recall versus false events per image", description: "Whole-data-set and test-split recall against false events per image for the on-board cascade", name: "full sweep" } })] }));
kids.push(note("Figure 1. Recall against false events per image for the on-board cascade, sweeping the CNN threshold over the stored logits (blue: all ships; red dashed: inshore ships); black points are the four validation-calibrated operating points with image-bootstrap 95 % CIs. The Weibull-only system sits at 99.6-99.7 % recall and 166-173 false events per image, far to the right of the plotted range."));
const A97 = fsr("all").find((r) => r.system.includes("97 %")), T97 = fsr("test").find((r) => r.system.includes("97 %")), WA = fsr("all")[0];
kids.push(P([B("Reading Table 2a. "), "The prescreen alone finds ", pc(WA.recall), " % of the 16,951 ships (inshore ", pc(WA.recall_inshore), " %, offshore ", pc(WA.recall_offshore), " %) but leaves ", f1(WA.FA_per_img), " false events per image. The CNN removes about ", (100 * (1 - A97.FA_per_img / WA.FA_per_img)).toFixed(1), " % of them at the 97 % target: ", pc(A97.recall), " % recall at ", f2(A97.FA_per_img), " false events per image on all images, ", pc(T97.recall), " % at ", f2(T97.FA_per_img), " on the held-out test split. The ", f1(100 * (WA.recall - A97.recall)), "-point recall cost on all images is the price of the CNN stage. The test split's confidence intervals are about three times wider than those of the whole data set, and the train, validation and test recalls agree within their intervals at every operating point, so the in-sample rows are not visibly inflated."]));

kids.push(H1("5b. Training statistics: seeds and splits"));
kids.push(P(["The CNN numbers so far came from one seed and one split. Four further INT8 trainings were run for each network (context and single tower) with the identical recipe: two more seeds on the original split (sd2, sd3) and two more splits (split seeds 20261004 and 20261005, training seed 1; sp2, sp3), each with its own validation thresholds and its own held-out test images. Together with the original run (sd1) this gives five runs per network. Table 2b gives mean and standard deviation over the runs, and the image-bootstrap 95 % CI of the original run for reference; Table 2c compares the two networks run by run."]));
kids.push(caption("Table 2b. Test-split results at the validation-calibrated thresholds, mean +- s.d. over the runs (n = number of runs)."));
const T7 = SS.map((r) => [r.model === "context" ? "Weibull + context CNN" : "Weibull + single-tower CNN", pc(r.target) + " %", r.n_runs, pc(r.recall_mean) + " +- " + pc(r.recall_sd), pc(r.recall_inshore_mean) + " +- " + pc(r.recall_inshore_sd), f2(r.FA_per_img_mean) + " +- " + f2(r.FA_per_img_sd), f3(r.F1_mean), "[" + pc(r.recall_boot_lo) + ", " + pc(r.recall_boot_hi) + "]", "[" + f2(r.FA_per_img_boot_lo) + ", " + f2(r.FA_per_img_boot_hi) + "]"]);
kids.push(table([3100, 1000, 700, 2200, 2200, 2100, 800, 1700, 1700], ["Network", "Val. retention target", "n", "Recall % (mean +- s.d.)", "Inshore recall % (mean +- s.d.)", "False events / img (mean +- s.d.)", "F1", "Run sd1: bootstrap 95 % CI, recall %", "Run sd1: bootstrap 95 % CI, FA / img"], T7, 15));
kids.push(note("Splits sp2 and sp3 hold out different images, so the s.d. includes split-to-split variation in the test set. Per-run values: Results/fixedpoint/seeds_all_runs.csv; s.d. over the three seeds of the original split only: seeds_summary.csv."));
kids.push(caption("Table 2c. Context network minus single-tower network, paired by run (same split and seed)."));
const T8 = SPR.map((r) => [pc(r.target) + " %", r.n_pairs, (100 * r.dRecall_mean).toFixed(2) + " (" + (100 * r.dRecall_min).toFixed(2) + " to " + (100 * r.dRecall_max).toFixed(2) + ")", (100 * r.dInshore_mean).toFixed(2), pc(r.FA_reduction_mean) + " (" + pc(r.FA_reduction_min) + " to " + pc(r.FA_reduction_max) + ")", r.ctx_fewer_FA_in + " of " + r.n_pairs]);
kids.push(table([1800, 800, 3800, 2400, 3800, 2200], ["Retention target", "Pairs", "Recall difference, points: mean (range)", "Inshore recall, points: mean", "False-event reduction %: mean (range)", "Runs with fewer false events"], T8, 15));
kids.push(P([B("What this changes in the earlier claim. "), "The single-run comparison suggested that the context network gains about two points of recall at equal false events. Over the runs the paired result is different and replaces it: at the validation-calibrated thresholds the two networks keep nearly the same fraction of ships (the context network is ahead by under half a point on average and behind in some runs), while the context network produces fewer false events in every paired run, ", pc(SPR[1].FA_reduction_mean), " % fewer at the 95 % target and ", pc(SPR[2].FA_reduction_mean), " % fewer at the 97 % target on average. The context tower therefore buys precision at fixed recall, not recall at fixed false events. The run-to-run spread of recall at a fixed target is about one point (s.d.), larger than the recall difference between the two networks, so a recall difference cannot be claimed; the false-event difference can."]));

kids.push(H1("6. Model cost: where the cascade differs from the detectors"));
kids.push(P("The point of the cascade is cost, not AP. The prescreen is a fixed integer datapath with no learned parameters (a 25 x 25 ring sum, a multiplier-based Weibull test and a 5 x 5 peak search per pooled pixel); the CNN sees only the candidate events (about 181 per image on average, median 33, at most 2,616) as 32 x 32 patches of the pooled image."));
kids.push(caption("Table 3. Model size and compute of the proposed CNN against the lightweight detectors reported in Table 1."));
kids.push(table([3600, 2400, 2800, 3200, 3000], ["Model", "Parameters", "Compute", "Weights in memory", "Note"], [
  ["Proposed INT8 CNN (per candidate)", "65,633", "1.95 M MAC per 32 x 32 patch", "about 64 KiB int8 + biases", "mean 181 candidates / image -> 0.35 GMAC (0.71 GFLOP) per 800 x 800 frame; median image 33 candidates -> 0.06 GMAC"],
  ["Proposed INT8 context CNN (per candidate)", "74,609 (fine tower 14,304; context tower 3,696; side 160; head 56,449)", "2.31 M MAC per candidate (fine 1.90 M, context 0.36 M, head 0.06 M)", "about 73 KiB int8 + biases", "mean 181 candidates / image -> 0.42 GMAC per 800 x 800 frame; plus a 4,096-read context-patch fetch per candidate"],
  ["Proposed prescreen", "none (constants only)", "integer adds + 3 multiplies per pooled pixel", "frame store 160 kB (shared with the CNN patch fetch)", "no learned weights; Pfa selects one of four constants"],
  ["AC-YOLO [11]", "1.8 M", "5.4 GFLOPs", "3.75 MB", "input size not stated in the consulted text"],
  ["LH-YOLO [10]", "1.862 M", "-23.8 % vs YOLOv8n", "n.r.", "FLOPs absolute value n.r."],
  ["MEA-Net [5]", "0.96 M", "2.80 GFLOPs", "n.r.", "Jetson Nano, 6.31 FPS"],
  ["LRTransDet [6]", "3.07 M", "n.r.", "n.r.", "75.8 FPS on HRSID"],
  ["LMSD-YOLO [4]", "n.r.", "n.r.", "7.6 MB", "Jetson AGX Xavier, 68.3 FPS"],
  ["HRSDNet [1]", "n.r.", "n.r.", "728.2 MB", "0.154 s / image, RTX 2080"],
], 16));
kids.push(note("MAC / FLOP conventions differ between papers (1 MAC = 2 FLOPs here: 1.95 M MAC = 3.9 MFLOP per patch). Our per-frame figure counts only candidate patches; image-level pooling and the prescreen are integer additions and are not included. The lightweight detectors are 15-47 times larger in parameters than our CNN, but they also produce boxes, classify and localise without a separate CFAR stage."));

kids.push(H1("7. Hardware comparison"));
kids.push(P("Table 4 follows Table VIII of [2] and adds the HRSID-relevant implementations found in the literature and in the project vault. The last two rows are measured: the prescreen alone (Quartus 21.1 fit of the 800 x 800 design including the 160 kB frame store, time measured on the board) and the full cascade on the DE10-Standard (fit, and timing over 1,000 images)."));
kids.push(caption("Table 4. FPGA / edge implementations (layout of Table VIII in [2])."));
kids.push(table([1150, 2700, 1650, 1250, 1750, 1500, 1500, 1050, 950, 1550], ["Ref.", "Algorithm", "Platform", "Image size", "Detection accuracy", "Time", "Logic", "BRAM / M10K", "DSP", "Clock / power"], T4, 14, [7, 8]));
kids.push(note("[19] and [20] report accuracy on data sets other than HRSID (SAR-Ship, SSDD; xView3-SAR), and [2] on its own mixed set, so their accuracy entries are not comparable with HRSID numbers; they are included for the hardware metrics only. Entries for ref. [10]-[12] are copied from Table VIII of [2]. \"n.r.\" = not reported in the text consulted. Times of our design are read from the on-chip cycle counters during the on-board sweep (prescreen: (400+24)^2 x 5 = 900,513 clock cycles at 100 MHz; CNN: 19,658 clock cycles per candidate); n.m. = not measured. Our JTAG pixel upload (0.2-0.3 s per frame) is a host limitation and is not part of the quoted times; a streaming sensor interface would need 12.8 ms for the 640,000 pixels at 50 MHz."));
kids.push(caption("Table 5. Same-algorithm baselines in the manner of Table VII of [2]: the earlier streaming-Weibull cascade vs the present two-pass pooled cascade (same board, same JTAG feed, same 1,000 images; resources from Quartus fits at 800 x 800)."));
kids.push(table([3600, 3300, 2900, 2900, 2338], ["Quantity", "Earlier design (streaming 17/13 Weibull + DEEP INT8)", "This work: pooled prescreen + single-tower INT8", "This work: pooled prescreen + context INT8", "Context vs earlier"], T5, 15));
kids.push(note("Reduction = 1 - new/old. Both designs use the quad-pixel CNN core at 100 MHz; the CNN tables differ (DEEP trained on the old candidates vs the single-tower network retrained on config A candidates), so the time and accuracy rows combine the effect of the prescreen and of the CNN training. Power was not measured for either design. Clock-domain and handshake structure are the same in both."));
kids.push(P([B("Still to add. "), "(a) time per frame of the same software model on CPU and GPU, as in Fig. 13 of [2]; (b) power (not yet measured); (c) a real-time ratio in the sense of [2] for a stated sensor rate; (d) nothing further for the context-tower variant, which is now built, fitted and verified on the board (Tables 4 and 5). Done since the first version of this document: the 1,000-image on-board sweep of the new cascade (0 mismatches over 71,380 events; the earlier design had 0 mismatches over 725,893 events), so the hardware accuracy column equals the software model, as [2] shows for its detection rate."]));

kids.push(H1("8. Limitations and next steps for the software comparison"));
kids.push(bullet("Our test split is not the official HRSID split of [1]. A like-for-like row needs the cascade evaluated on the official 35 % test images (the prescreen has no training, so only the CNN needs retraining on the official training images)."));
kids.push(bullet("An IoU-based AP50 for our system would require box output; a cheap option is a threshold-and-grow step around each accepted event, evaluated against HRSID polygons. Until then, recall at a stated false-event rate is the honest metric."));
kids.push(bullet("Several literature numbers come from abstracts only; before submission each row should be re-read in its full text (precision, recall, F1, GFLOPs, FPS and platform), and the RLE-YOLO value checked."));
kids.push(bullet("The CNN statistics are five runs per network (three seeds on one split, two further splits); with five runs the s.d. is itself uncertain, and a larger repeat or cross-validation over the 5,604 images would tighten the recall intervals, which are currently about +-1 point."));
kids.push(bullet("Power is not measured for any of the designs, so the hardware comparison has no energy column of its own."));

kids.push(H1("Software evidence in this repository"));
kids.push(P("Prescreen golden model, whole-data-set evaluation: _comparison/fixedpoint/prescreen_fx.py, eval_fx_dataset.py, report_fx.py. Candidate extraction: extract_fx_events.py. CNN: _comparison/cnn/train_ctx.py, train_q.py, eval_pooldet.py. Table 2: software_metrics.py and prep_comparison_data.py (outputs in _comparison/Results/fixedpoint/). RTL: rtl/prescreen/. Design notes: PAPER2_FIXEDPOINT_RTL_PRESCREEN_2026-10-03.md."));

kids.push(H1("References"));
REFS.forEach((r) => kids.push(new Paragraph({ spacing: { after: 50, line: 250 }, alignment: AlignmentType.LEFT, children: [run(r, { size: 18 })] })));

const doc = new Document({
  creator: "Paper 2 working group", title: "HRSID literature comparison",
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 900, bottom: 900, left: 900, right: 900 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 18 })] })] }) }, children: kids }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
