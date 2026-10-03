# -*- coding: utf-8 -*-
"""Bit-exact integer model of the POOLED Weibull prescreen (Paper 2, config A family).  This file is the golden model
for rtl/prescreen/*.v : every quantity below is an integer and every operation is an add, a shift or a multiply.

Datapath (all integers):
  q8   = QROM[pixel]                                  256-entry ROM, 8 bit  (rtl/cascade/qrom.hex)
  P    = (q00+q01+q10+q11+2) >> 2                     2x2 pooled frame, 8 bit (the frame the CNN patch store already holds)
  Pp   = P padded by TK = (SLI-1)/2 with mirrored border (edge sample repeated, numpy 'symmetric')
  S1   = sum of Pp over the SLI x SLI window minus the GUARD x GUARD window          (N = SLI^2-GUARD^2 cells)
  S2   = same for Pp^2
  A    = N*P - S1                                     contrast x N   (signed)
  num  = N*S2 - S1^2                                  = N*(N-1)*c2   (>= 0)
  Weibull detection  x > c1 + K/C,  C = clamp(pi/sqrt(6 c2), CMIN, CMAX),  K = gamma + ln(-ln Pfa)
     <=>  A > 0  and  A^2 > Kc * clamp(num, NUM_LO, NUM_HI)     with Kc = kappa^2 N/(N-1), kappa = K sqrt(6)/pi
     (no ROM, no sqrt, no divider; the shape clamp becomes a clamp on num)
  gate   A >= G                                       (G = N * tau / STEP, i.e. contrast >= tau in log units)
  event  = gated detected pixel whose A is the maximum of its 5x5 neighbourhood (ties kept), outside the frame = -inf
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
XLO, XHI = -0.40, 2.80
STEP = (XHI - XLO) / 255.0                     # ln-amplitude per q8 code
EULER = 0.5772156649015329
PFAS = (3e-2, 1e-2, 1e-3, 1e-4)                # pfa_sel 0..3
CMIN, CMAX = 0.8, 8.0                          # Weibull shape clamp (same as the float model)


def qrom():
    t = np.zeros(256, np.int64)
    for i in range(256):
        x = 0.5 * math.log(i + 0.5)
        t[i] = math.floor(min(max((x - XLO) / (XHI - XLO), 0.0), 1.0) * 255 + 0.5)
    return t


QROM = qrom()


class Cfg:
    """Compile-time constants.  KF = fractional bits of Kc, SHN = right shift applied to the clamped num before the multiply."""

    def __init__(self, sli=25, guard=17, pfa_sel=0, tau=0.6, event=5, KF=12, SHN=12):
        self.sli, self.guard, self.pfa_sel, self.tau, self.event, self.KF, self.SHN = sli, guard, pfa_sel, tau, event, KF, SHN
        self.TK, self.TG = (sli - 1) // 2, (guard - 1) // 2
        self.N = sli * sli - guard * guard
        N = self.N
        K = EULER + math.log(-math.log(PFAS[pfa_sel]))
        self.kappa = K * math.sqrt(6) / math.pi
        self.Kc_ideal = self.kappa ** 2 * N / (N - 1)
        self.KC = int(round(self.Kc_ideal * 2 ** KF))
        smin = math.pi / (CMAX * math.sqrt(6)) / STEP            # sigma-hat range in q8 codes where the shape clamp is inactive
        smax = math.pi / (CMIN * math.sqrt(6)) / STEP
        self.NUM_LO = int(round(smin ** 2 * N * (N - 1)))
        self.NUM_HI = int(round(smax ** 2 * N * (N - 1)))
        self.G = int(round(N * tau / STEP))


def pool_q8(img):
    """img: HxW uint8 (cropped to even size by the caller or here).  Returns the 8-bit pooled frame P."""
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    q = QROM[img[:h, :w].astype(np.int64)]
    s = q[0::2, 0::2] + q[0::2, 1::2] + q[1::2, 0::2] + q[1::2, 1::2]
    return (s + 2) >> 2


def box(A, k):
    C = np.pad(np.cumsum(np.cumsum(A, 0), 1), ((1, 0), (1, 0)))
    return C[k:, k:] - C[:-k, k:] - C[k:, :-k] + C[:-k, :-k]


def ring_sums(P, cfg):
    TK, TG = cfg.TK, cfg.TG
    Pp = np.pad(P.astype(np.int64), TK, mode="symmetric")
    def ring(A):
        return box(A, cfg.sli) - box(A[TK - TG:A.shape[0] - (TK - TG), TK - TG:A.shape[1] - (TK - TG)], cfg.guard)
    return ring(Pp), ring(Pp * Pp)


def local_max(Cd, size):
    """size x size maximum filter, -inf (int64 min) outside the frame."""
    r = size // 2
    h, w = Cd.shape
    NEG = np.iinfo(np.int64).min
    Pd = np.full((h + 2 * r, w + 2 * r), NEG, np.int64)
    Pd[r:r + h, r:r + w] = Cd
    out = Pd[0:h, 0:w].copy()
    for dy in range(size):
        for dx in range(size):
            out = np.maximum(out, Pd[dy:dy + h, dx:dx + w])
    return out


def prescreen(P, cfg):
    """P: pooled 8-bit frame (int).  Returns dict of integer maps; E is the event mask."""
    S1, S2 = ring_sums(P, cfg)
    N = cfg.N
    A = N * P.astype(np.int64) - S1
    num = N * S2 - S1 * S1
    numc = np.clip(num, cfg.NUM_LO, cfg.NUM_HI)
    R = (numc >> cfg.SHN) * cfg.KC
    # A^2 (34 bit) is compared with R = Kc*numc * 2^(KF+SHN-SHN)... both sides carry the factor 2^(KF - 0): see note
    L = (A * A) if cfg.SHN == cfg.KF else None
    if L is None:
        raise ValueError("this model fixes SHN == KF so the two scale factors cancel")
    D = (A > 0) & (L > R)
    Dg = D & (A >= cfg.G)
    NEG = np.iinfo(np.int64).min
    Cd = np.where(Dg, A, NEG)
    E = Dg & (Cd >= local_max(Cd, cfg.event))
    return dict(S1=S1, S2=S2, A=A, num=num, numsh=(numc >> cfg.SHN), D=D, Dg=Dg, E=E)


def prescreen_float_on_q8(P, cfg):
    """Ideal-arithmetic reference on the SAME 8-bit data: exact sqrt/clamp/real Kc, no integer approximations."""
    S1, S2 = ring_sums(P, cfg)
    N = cfg.N
    A = N * P.astype(np.float64) - S1
    num = (N * S2 - S1 * S1).astype(np.float64)
    c2 = num / (N * (N - 1))
    sig = np.sqrt(np.maximum(c2, 0))
    smin = math.pi / (CMAX * math.sqrt(6)) / STEP
    smax = math.pi / (CMIN * math.sqrt(6)) / STEP
    thr = cfg.kappa * np.clip(sig, smin, smax)
    D = A / N > thr
    Dg = D & (A / N >= cfg.tau / STEP)
    Cd = np.where(Dg, A, -np.inf)
    h, w = Cd.shape
    r = cfg.event // 2
    Pd = np.full((h + 2 * r, w + 2 * r), -np.inf); Pd[r:r + h, r:r + w] = Cd
    mx = Pd[0:h, 0:w].copy()
    for dy in range(cfg.event):
        for dx in range(cfg.event):
            mx = np.maximum(mx, Pd[dy:dy + h, dx:dx + w])
    return dict(D=D, Dg=Dg, E=Dg & (Cd >= mx))


def events_rowmajor(E):
    ys, xs = np.nonzero(E)
    return np.stack([ys, xs], 1)
