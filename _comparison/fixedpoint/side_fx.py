# -*- coding: utf-8 -*-
"""Integer definition of the 9 side features of the context CNN (golden model for the RTL side-feature unit).  All are uint8 codes.

Per event (from the prescreen: A = N*(P-c1), S1 = ring sum of P, numsh = clamp(num)>>12, P = pooled amplitude):
  f0 = min(255, (A*195 + 2^15) >> 16)            contrast in q8 codes (195/65536 ~ 1/N, N = 336)
  f1 = (S1*195 + 2^15) >> 16                      local ring mean (q8 code)
  f2 = isqrt((numsh*2385 + 2^15) >> 16)           local ring sigma (q8 code; 2385/65536 ~ 4096/(N(N-1)))
  f3 = P                                          event amplitude (q8 code)
Per image (accumulated while the frame streams in; Q = QROM code of every full-resolution pixel, NPIX = W*H, R = round(2^40/NPIX)):
  f4 = (sumQ*R + 2^39) >> 40                      image mean of Q
  f5 = isqrt(max(0, ((sumQ2*R + 2^39) >> 40) - f4^2))   image standard deviation of Q
  f6 = min(255, (cntBright*R + 2^28) >> 29)       fraction of pixels > 200, x2048
  f7 = min(255, (cntDark*R*255 + 2^39) >> 40)     fraction of pixels < 5, x255
  f8 = LN[nE] = min(255, round(25*ln(1+nE)))      nE = number of events of the frame (ROM, nE <= 8191)
"""
import math

import numpy as np

K_A, K_NUM = 195, 2385
LN = np.array([min(255, int(math.floor(25 * math.log(1 + n) + 0.5))) for n in range(8192)], np.int64)


def isqrt_arr(v):
    v = np.asarray(v, np.int64)
    r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64)
    r = np.where((r + 1) * (r + 1) <= v, r + 1, r)
    r = np.where(r * r > v, r - 1, r)
    return r


def event_codes(A, S1, numsh, P):
    A, S1, numsh, P = (np.asarray(x, np.int64) for x in (A, S1, numsh, P))
    f0 = np.minimum(255, (A * K_A + (1 << 15)) >> 16)
    f1 = (S1 * K_A + (1 << 15)) >> 16
    f2 = isqrt_arr((numsh * K_NUM + (1 << 15)) >> 16)
    return np.stack([f0, f1, f2, P], 1)


def image_codes(Q, I, n_events):
    """Q: QROM codes of all full-resolution pixels (2-D int array), I: raw pixels, n_events: events of the frame."""
    npix = Q.size
    R = int(round(2 ** 40 / npix))
    sumQ = int(Q.sum()); sumQ2 = int((Q.astype(np.int64) ** 2).sum()); cB = int((I > 200).sum()); cD = int((I < 5).sum())
    f4 = (sumQ * R + (1 << 39)) >> 40
    var = max(0, ((sumQ2 * R + (1 << 39)) >> 40) - f4 * f4)
    f5 = int(isqrt_arr(var))
    f6 = min(255, (cB * R + (1 << 28)) >> 29)
    f7 = min(255, (cD * R * 255 + (1 << 39)) >> 40)
    f8 = int(LN[min(n_events, 8191)])
    return np.array([f4, f5, f6, f7, f8], np.int64)
