# -*- coding: utf-8 -*-
"""256-entry pixel -> q8 ROM: q = floor(clip((log(sqrt(I+0.5)) - XLO)/(XHI-XLO), 0, 1)*255 + 0.5).
Identical to the quantiser the CNN's training patches were built with (extract_cnn_patches_hwspec.m)."""
import math, sys
XLO, XHI = -0.40, 2.80
out = sys.argv[1] if len(sys.argv) > 1 else "qrom.hex"
with open(out, "w") as f:
    for i in range(256):
        x = 0.5 * math.log(i + 0.5)
        q = math.floor(min(max((x - XLO) / (XHI - XLO), 0.0), 1.0) * 255 + 0.5)
        f.write("%02x\n" % q)
print("wrote", out)
