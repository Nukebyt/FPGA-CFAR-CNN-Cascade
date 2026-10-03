# -*- coding: utf-8 -*-
"""Crop a square test image from an HRSID scene (centred on the first ship) -> img.hex (one byte per line)."""
import json, sys, os
import numpy as np
from PIL import Image

root = r"F:\Projects\CFAR\HRSID"
size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
out = sys.argv[2] if len(sys.argv) > 2 else "img.hex"
which = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ann = json.load(open(os.path.join(root, "annotations", "train_test2017.json")))
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a["bbox"])
imgs = {im["id"]: im["file_name"] for im in ann["images"]}
cands = [i for i in sorted(by_img) if len(by_img[i]) >= 2][which:which + 1]
iid = cands[0]
im = np.array(Image.open(os.path.join(root, "images", imgs[iid])))
if im.ndim == 3:
    im = im[:, :, 0]
x, y, w, h = by_img[iid][0]
cx, cy = int(x + w / 2), int(y + h / 2)
x0 = min(max(cx - size // 2, 0), im.shape[1] - size); y0 = min(max(cy - size // 2, 0), im.shape[0] - size)
x0 -= x0 % 2; y0 -= y0 % 2
crop = im[y0:y0 + size, x0:x0 + size].astype(np.uint8)
with open(out, "w") as f:
    for v in crop.reshape(-1):
        f.write("%02x\n" % v)
print("image", imgs[iid], "crop", (y0, x0), "size", crop.shape, "->", out)
