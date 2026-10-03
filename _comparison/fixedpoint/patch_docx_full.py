# -*- coding: utf-8 -*-
"""one-off patch of build_comparison_docx.js: whole-data-set board sweep (5a), seeds/splits (5b), stale statements"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "build_comparison_docx.js")
s = open(p, encoding="utf-8").read()
blk = open(os.path.join(HERE, "docx_block_full.js"), encoding="utf-8").read()
if "5a. Whole data set" in s:
    raise SystemExit("already patched")
a = "const pre = Q8[0];"
s = s.replace(a, a + '\nconst FS = csv("full_ctx_metrics.csv"), SS = csv("seeds_summary.csv"), SPR = csv("seeds_paired.csv");\nconst ci = (v, lo, hi, g) => g(v) + " (" + g(lo) + "-" + g(hi) + ")";\nconst fsr = (sub) => FS.filter((r) => r.subset === sub);', 1)
m = 'kids.push(H1("6. Model cost: where the cascade differs from the detectors"));'
assert m in s
s = s.replace(m, blk + m, 1)
s = s.replace("Footer, PageNumber, PageOrientation }", "Footer, PageNumber, PageOrientation, ImageRun }", 1)
old1 = 'kids.push(bullet("One seed and one split for the CNN results; confidence intervals (bootstrap over images) are still to be computed."));'
new1 = 'kids.push(bullet("The CNN statistics are five runs per network (three seeds on one split, two further splits); with five runs the s.d. is itself uncertain, and a larger repeat or cross-validation over the 5,604 images would tighten the recall intervals, which are currently about +-1 point."));'
old2 = 'kids.push(bullet("The context-tower model is a software result only; its hardware cost (second tower, second 4x4-pooled store, side-feature unit) has not been evaluated."));'
new2 = 'kids.push(bullet("Power is not measured for any of the designs, so the hardware comparison has no energy column of its own."));'
for o, n in ((old1, new1), (old2, new2)):
    assert o in s; s = s.replace(o, n, 1)
o3 = "(d) the context-tower variant. Done since"
assert o3 in s
s = s.replace(o3, "(d) nothing further for the context-tower variant, which is now built, fitted and verified on the board (Tables 4 and 5). Done since", 1)
o4 = "(unchanged at the 95 % target: \", pc(cx(0.95).recall_inshore), \" %).\"]));"
assert o4 in s
s = s.replace(o4, "(unchanged at the 95 % target: \", pc(cx(0.95).recall_inshore), \" %). This is the original run; the paired comparison over five runs in Table 2c shows the robust effect is fewer false events at about equal recall, not the two points of recall.\"]));", 1)
open(p, "w", encoding="utf-8").write(s); print("patched")
