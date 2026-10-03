# -*- coding: utf-8 -*-
"""Host side of the JTAG-fed cascade with the pooled-domain prescreen (cascade_ps_jtag_top; register map in rtl/prescreen/jtag/cascade_ps_jtag_core.v).

    from board_ps import BoardPS
    b = BoardPS()
    r = b.run_frame(pixels_uint8_800x800, pfa_sel=0, g_th=16065, theta=-17457)

pfa_sel: 0 = 3e-2, 1 = 1e-2, 2 = 1e-3, 3 = 1e-4.   g_th = round(336 * tau / 0.012549) (tau 0.6 -> 16065).   theta = integer-logit threshold.
"""
import time

import numpy as np

from board import Board, A_CTRL, A_CFG, A_TAU, A_THETA, A_NEV, A_NACC, A_NRES, A_PIX, A_POST, A_RES

A_PSCYC = 0xB
LOGIT_SCALE = 0.00021863886012347044           # INT8 single-tower model pf_plain_q8


class BoardPS(Board):
    ID = 0xCA5CADE3

    def run_frame(self, pixels, pfa_sel=0, g_th=16065, theta=-(2 ** 31) + 1, timeout=120.0):
        px = np.ascontiguousarray(pixels, dtype=np.uint8)
        assert px.shape == (self.h, self.w), px.shape
        self.wr(A_CFG, pfa_sel); self.wr(A_TAU, g_th); self.wr(A_THETA, theta)
        t0 = time.time()
        while not self.status()["ready"]:
            if time.time() - t0 > timeout:
                raise TimeoutError("core not ready")
            time.sleep(0.01)
        self.wr(A_CTRL, 1)
        t1 = time.time()
        self.wr_pixels(px.tobytes())
        t2 = time.time()
        while not self.status()["done"]:
            if time.time() - t2 > timeout:
                raise TimeoutError("frame_done never asserted")
            time.sleep(0.005)
        st = self.status()
        nev, nacc, nres, pix, post, psc = (self.rd(a)[0] for a in (A_NEV, A_NACC, A_NRES, A_PIX, A_POST, A_PSCYC))
        words = self.rd(A_RES, 2 * nres) if nres else []
        lo = np.array(words[0::2], dtype=np.int64); hi = np.array(words[1::2], dtype=np.int64)
        hi = np.where(hi >= 1 << 31, hi - (1 << 32), hi)
        return dict(n_events=nev, n_accepted=nacc, n_res=nres, pix_issued=pix, post_cycles=post, ps_cycles=psc, status=st,
                    j=(lo >> 10) & 0x3FF, i=lo & 0x3FF, accept=(lo >> 29) & 1, logit=hi,
                    t_upload_s=t2 - t1, t_total_s=time.time() - t0)
