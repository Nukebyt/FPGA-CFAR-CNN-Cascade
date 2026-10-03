# -*- coding: utf-8 -*-
"""RTL inputs for cnn_core_ctx.v (fine tower + context tower + side MLP + concatenated head, one layer-sequential program) from the
integer model written by _comparison/cnn/train_q3.py (<name>_int3.pt).

Layer program (9 layers; A/B = the two activation memories, 4 parity banks each):
   0 S   side fc 9->16          A@SIDE_BASE (9 ch, 1x1)           -> B@SIDE_OUT
   1 C0  conv5 1->8  +pool      A@CTX_BASE (ctx patch 32x32)      -> B@0
   2 C1  conv3 8->16 +pool      B@0                               -> A@C1_BASE
   3 C2  conv3 16->16 +pool     A@C1_BASE                         -> B@C2_BASE (16 ch, 2x2)
   4 F0  conv5 1->16 +pool      A@0 (fine patch 32x32)            -> B@0
   5 F1  conv3 16->32           B@0                               -> A@0
   6 F2  conv3 32->32 +pool     A@0                               -> B@0   (32 ch, 5x5)
   7 H1  fc (800+64+16)->64     B@0 (K5, 32 ch) + B@C2_BASE (K2, 16 ch) + B@SIDE_OUT (K1, 16 ch)  -> A@0     three SEGMENTS summed in the accumulator
   8 H2  fc 64->1               A@0                                -> logit
The three feature segments of H1 share one activation scale (tied in QAT), so their raw integer partial sums add.
usage: python gen_cnn_rtl_ctx.py <int3.pt> <out_dir> [--lanes 32]"""
import argparse
import json
import math
import os

import numpy as np
import torch

CTX_BASE, C1_BASE, SIDE_BASE = 256, 512, 1152
C2_BASE, SIDE_OUT = 1152, 1168


def cdiv2(v):
    return (v + 1) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("int3"); ap.add_argument("out_dir"); ap.add_argument("--lanes", type=int, default=32)
    a = ap.parse_args()
    L = a.lanes
    I = torch.load(a.int3, weights_only=False)["ints3"]
    nin = 32
    # (name, layer dict, input segments [(cin, hin, base)], imem, omem, obase)
    prog = [
        ("S",  I["side"],    [(9, 1, SIDE_BASE)],  0, 1, SIDE_OUT),
        ("C0", I["ctx"][0],  [(1, nin, CTX_BASE)], 0, 1, 0),
        ("C1", I["ctx"][1],  [(8, 14, 0)],         1, 0, C1_BASE),
        ("C2", I["ctx"][2],  [(16, 6, C1_BASE)],   0, 1, C2_BASE),
        ("F0", I["fine"][0], [(1, nin, 0)],        0, 1, 0),
        ("F1", I["fine"][1], [(16, 14, 0)],        1, 0, 0),
        ("F2", I["fine"][2], [(32, 12, 0)],        0, 1, 0),
        ("H1", I["h1"],      [(32, 5, 0), (16, 2, C2_BASE), (16, 1, SIDE_OUT)], 1, 0, 0),
        ("H2", I["h2"],      [(64, 1, 0)],         0, 0, 0),
    ]
    layers, wwords, pqwords = [], [], []
    wbase = pbase = 0
    bank_depth = 0
    for name, ent, segs, imem, omem, obase in prog:
        W = np.asarray(ent["W"], np.int64); bias = np.asarray(ent["b"], np.int64)
        relu, pool = int(ent["relu"]), int(ent["pool"])
        cout = W.shape[0]
        seginfo = []
        T = 0
        for si, (cin, hin, base) in enumerate(segs):
            if ent["kind"] == "conv":
                k = W.shape[2]; assert len(segs) == 1 and W.shape[1] == cin
            else:                                                    # fc = conv whose kernel covers the whole incoming map
                k = hin
            seginfo.append(dict(K=k, CIN=cin, WB=cdiv2(hin), BSZ=cdiv2(hin) * cdiv2(hin), BASE=base))
            T += cin * k * k
        assert W.reshape(cout, -1).shape[1] == T, (name, W.shape, T)
        k0, hin0 = seginfo[0]["K"], segs[0][1]
        hout = hin0 - k0 + 1
        if ent["kind"] == "fc":
            hout = 1
        if hout == 1:
            mode, qh, hmap = 0, 1, 1
        elif pool:
            assert hout % 2 == 0; mode, qh, hmap = 2, hout // 2, hout // 2
        else:
            assert hout % 2 == 0; mode, qh, hmap = 1, hout // 2, hout
        wb_o = cdiv2(hmap); bsz_o = wb_o * cdiv2(hmap)
        G = math.ceil(cout / L)
        Wf = W.reshape(cout, -1)
        for g in range(G):
            for t in range(T):
                word = 0
                for l in range(L):
                    ch = g * L + l
                    v = int(Wf[ch, t]) if ch < cout else 0
                    word |= (v & 0xFF) << (8 * l)
                wwords.append(word)
        M = np.asarray(ent["M"], np.int64) if relu else np.zeros(cout, np.int64)
        for g in range(G):
            for l in range(L):
                ch = g * L + l
                pqwords.append((((int(bias[ch]) if ch < cout else 0) & 0xFFFFFFFF) << 16) | ((int(M[ch]) if ch < cout else 0) & 0xFFFF))
        s0 = seginfo[0]
        layers.append(dict(name=name, K=s0["K"], Cin=s0["CIN"], Cout=cout, T=T, G=G, relu=relu, S=int(ent.get("S", 0) or 0), WBASE=wbase, PBASE=pbase,
                           MODE=mode, QH=qh, QW=qh, WB_IN=s0["WB"], BSZ_IN=s0["BSZ"], WB_O=wb_o, BSZ_O=bsz_o,
                           IMEM=imem, OMEM=omem, IBASE=s0["BASE"], OBASE=obase, NSEG=len(segs), segs=seginfo))
        wbase += G * T; pbase += G * L
        if hout != 1 or relu:
            bank_depth = max(bank_depth, obase + cout * bsz_o)
        for sg in seginfo:
            bank_depth = max(bank_depth, sg["BASE"] + sg["CIN"] * sg["BSZ"])
    assert layers[-1]["relu"] == 0 and layers[-1]["Cout"] == 1
    aw = max(1, math.ceil(math.log2(bank_depth)))
    waw = max(1, math.ceil(math.log2(wbase))); pqaw = max(1, math.ceil(math.log2(pbase)))
    tw = max(1, math.ceil(math.log2(max(l["T"] for l in layers) + 1)))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "cnn_w.hex"), "w") as f:
        for w in wwords:
            f.write(format(w, "0%dx" % (L * 2)) + "\n")
    with open(os.path.join(a.out_dir, "cnn_pq.hex"), "w") as f:
        for w in pqwords:
            f.write(format(w, "012x") + "\n")
    fields = ["K", "Cin", "Cout", "T", "G", "relu", "S", "WBASE", "PBASE", "MODE", "QH", "QW", "WB_IN", "BSZ_IN", "WB_O", "BSZ_O", "IMEM", "OMEM", "IBASE", "OBASE", "NSEG"]
    lines = ["// generated by gen_cnn_rtl_ctx.py -- do not edit",
             f"localparam integer NL = {len(layers)};", f"localparam integer NIN = {nin};",
             f"localparam integer BANK_DEPTH = {bank_depth};", f"localparam integer AW = {aw};",
             f"localparam integer WDEPTH = {wbase};", f"localparam integer WAW = {waw};",
             f"localparam integer PQDEPTH = {pbase};", f"localparam integer PQAW = {pqaw};",
             f"localparam integer TW = {tw};", f"localparam integer LANES_GEN = {L};",
             f"localparam integer WB_NIN = {cdiv2(nin)};",
             f"localparam integer CTX_BASE = {CTX_BASE};", f"localparam integer SIDE_BASE = {SIDE_BASE};", f"localparam integer N_SIDE = 9;"]
    for fld in fields:
        lines.append(f"function integer L_{fld}; input integer l; begin case (l)")
        for i, ly in enumerate(layers):
            lines.append(f"    {i}: L_{fld} = {ly[fld]};")
        lines.append(f"    default: L_{fld} = 0; endcase end endfunction")
    for fld in ("K", "CIN", "WB", "BSZ", "BASE"):
        lines.append(f"function integer SEG_{fld}; input integer l; input integer s; begin case (l*4+s)")
        for i, ly in enumerate(layers):
            for si, sg in enumerate(ly["segs"]):
                lines.append(f"    {i*4+si}: SEG_{fld} = {sg[fld]};")
        lines.append(f"    default: SEG_{fld} = 0; endcase end endfunction")
    open(os.path.join(a.out_dir, "cnn_cfg.vh"), "w").write("\n".join(lines) + "\n")
    est = sum(l["G"] * l["QH"] * l["QW"] * l["T"] for l in layers)
    json.dump(dict(lanes=L, layers=[{k: v for k, v in l.items()} for l in layers], est_issue_cycles=est, bank_depth=bank_depth), open(os.path.join(a.out_dir, "cnn_params.json"), "w"), indent=1)
    print(f"{len(layers)} layers, L={L}: bank depth {bank_depth} x4 banks x2 buffers, ~{est} MAC-issue cycles/candidate")
    for l in layers:
        print(f"  {l['name']}: mode={l['MODE']} K={l['K']} Cin={l['Cin']} Cout={l['Cout']} T={l['T']} G={l['G']} quads={l['QH']}x{l['QW']} nseg={l['NSEG']} imem={l['IMEM']}@{l['IBASE']} omem={l['OMEM']}@{l['OBASE']}")


if __name__ == "__main__":
    main()
