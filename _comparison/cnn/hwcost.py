# -*- coding: utf-8 -*-
"""Cyclone V (5CSXFC6D6F31C6, DE10-Standard) budget model for the CNN stage.

MEASURED inputs (Quartus 21.1 fit reports in _quartus/weibull/):
    Weibull prescreen @ SLI=17/GUARD=13: 17,373 ALM, 22,507 regs, 13 DSP,
    309,183 block-mem bits, 70 M10K  -- at IMG_WIDTH=64.
DERIVED (this file, not a Quartus run -- verify with a fit before relying on it):
    * the 32 row-delay RAMs (16 rows x {16-bit x, 26-bit x^2}) scale with
      IMG_WIDTH; the other 38 M10K do not.
    * patch row buffer, patch ping-pong, CNN weight/activation M10K.
    * MAC-array throughput from lane count and a utilisation assumption.
DEVICE (Cyclone V handbook): 41,910 ALM; 553 M10K x 10,240 b; 112 DSP blocks,
    each = 3 independent 9x9 multipliers (an int8 x uint8 MAC fits a 9x9).
"""
import math
import sys

import hwlib as H

ALM_TOTAL, M10K_TOTAL, DSP_TOTAL = 41910, 553, 112
M10K_BITS = 10240
CLK = 50e6
WEIBULL = dict(alm=17373, dsp=13, m10k_fixed=38, rows=16)   # m10k_fixed = 70 - 32 row-delay RAMs @W=64
ASPECTS = [(8192, 1), (4096, 2), (2048, 5), (1024, 10), (512, 20), (256, 40)]


def m10k_for(depth, width):
    """Smallest M10K count for a depth x width RAM over the legal aspect ratios."""
    return min(math.ceil(depth / D) * math.ceil(width / Wd) for D, Wd in ASPECTS)


def weibull_m10k(img_w):
    rows = WEIBULL["rows"]
    return WEIBULL["m10k_fixed"] + rows * (m10k_for(img_w, 16) + m10k_for(img_w, 26))


def layer_table(cfg):
    size = cfg["size"] // cfg.get("down", 1)
    convs = H.parse_convs(cfg["convs"])
    fcs = [int(v) for v in cfg["fcs"].split(",")] if cfg["fcs"] else []
    rows, c, s = [], 1, size
    for (k, co, pool) in convs:
        so = s - k + 1
        rows.append(dict(name=f"conv{k}x{k}-{co}", macs=so * so * co * c * k * k, params=co * c * k * k + co,
                         in_b=c * s * s, out_b=co * so * so))
        c, s = co, so
        if pool:
            s //= 2
    d = c * s * s
    for h in fcs + [1]:
        rows.append(dict(name=f"fc-{h}", macs=d * h, params=d * h + h, in_b=d, out_b=h))
        d = h
    return rows


def report(name, cfg, img_w=800, lanes=128, util=0.7, cand_per_img=165.0, img_px=800 * 800, verbose=True, patch_px=None):
    rows = layer_table(cfg)
    macs = sum(r["macs"] for r in rows)
    params = sum(r["params"] for r in rows)
    P = patch_px or cfg["size"]
    # --- memory (M10K) ---
    wbl = weibull_m10k(img_w)
    patch_rows = m10k_for(img_w, 8) * P                  # P rows of 8-bit x, one RAM per row
    det_delay = math.ceil((P // 2) * img_w / 8192)       # 1-bit detection plane for the trigger delay
    pingpong = 2 * m10k_for(P * P, 8)
    wbits = params * 8
    wblk = math.ceil(wbits / M10K_BITS)                  # dense packing (optimistic; +20% for port widths)
    wblk = math.ceil(wblk * 1.2)
    act = max(r["in_b"] + r["out_b"] for r in rows) * 8
    ablk = 2 * math.ceil(act / M10K_BITS)
    m10k = wbl + patch_rows + det_delay + pingpong + wblk + ablk
    # --- compute ---
    dsp = 13 + math.ceil(lanes / 3) + 2                  # +2: requant multiplier(s)
    alm = WEIBULL["alm"] + 3000 + 60 * lanes + 1500      # trigger/ctrl + per-lane acc/mux + sequencer (estimate)
    cyc = sum(math.ceil(r["macs"] / (lanes * util)) for r in rows) + 200
    pps = CLK / cyc
    fps = pps / cand_per_img
    out = dict(name=name, params=params, macs=macs, m10k=m10k, m10k_pct=100 * m10k / M10K_TOTAL,
               dsp=dsp, dsp_pct=100 * dsp / DSP_TOTAL, alm=alm, alm_pct=100 * alm / ALM_TOTAL,
               cycles=cyc, patches_per_s=pps, frames_per_s=fps, mpix_per_s=fps * img_px / 1e6,
               parts=dict(weibull_rows=wbl, patch_rows=patch_rows, det_delay=det_delay, pingpong=pingpong,
                          weights=wblk, act=ablk))
    if verbose:
        print(f"{name}: {params:,} params, {macs/1e6:.2f}M MAC/patch | M10K {m10k} ({out['m10k_pct']:.0f}%) "
              f"DSP {dsp} ({out['dsp_pct']:.0f}%) ALM~{alm:,} ({out['alm_pct']:.0f}%) | "
              f"{cyc:,} cyc/patch -> {pps/1e3:.1f}k patch/s -> {fps:.1f} img/s ({out['mpix_per_s']:.1f} Mpx/s) "
              f"@ {lanes} lanes, {cand_per_img:.0f} cand/img")
    return out


if __name__ == "__main__":
    print("Weibull prescreen M10K vs image width (SLI=17):")
    for w in (64, 256, 512, 800, 1024):
        print(f"  W={w:4d}: {weibull_m10k(w):3d} M10K ({100*weibull_m10k(w)/M10K_TOTAL:.0f}%)")
