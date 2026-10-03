# -*- coding: utf-8 -*-
"""Host side of the JTAG-fed cascade (cascade_jtag_top).  Talks to jtag_server.tcl running inside System Console.

    from board import Board
    b = Board()                       # starts System Console + the server if needed
    b.run_frame(pixels_uint8_800x800, pfa_sel=0, tau_q14=12288, theta=-2**31+1)  ->  dict(events..., logits..., timing)

Register map: see rtl/cascade/jtag/cascade_jtag_core.v.
"""
import os
import socket
import subprocess
import tempfile
import time

import numpy as np

SYSCON = r"C:\intelFPGA_lite\21.1\quartus\sopc_builder\bin\system-console.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 5555
ID_EXPECT = 0xCA5CADE2
WIN = 0x40000                      # byte address of the pixel window
A_ID, A_CTRL, A_CFG, A_TAU, A_THETA, A_STATUS = 0x0, 0x1, 0x2, 0x3, 0x4, 0x5
A_NEV, A_NACC, A_NRES, A_PIX, A_POST = 0x6, 0x7, 0x8, 0x9, 0xA
A_RES = 0x1000
LOGIT_SCALE = 0.0001707530151151687          # real logit = int logit * scale (DEEP manifest)


class Board:
    ID = ID_EXPECT

    def __init__(self, port=PORT, start=True, w=800, h=800):
        self.port, self.w, self.h = port, w, h
        self.proc = None
        if start and not self._can_connect():
            env = dict(os.environ)
            self.proc = subprocess.Popen([SYSCON, "--script=" + os.path.join(HERE, "jtag_server.tcl"), str(port)],
                                         stdout=open(os.path.join(HERE, "syscon.log"), "w"), stderr=subprocess.STDOUT, cwd=HERE, env=env)
            for _ in range(120):
                if self._can_connect():
                    break
                time.sleep(1)
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=600)
        self.f = self.sock.makefile("rw", newline="\n")
        ident = self.rd(A_ID)[0]
        if ident != self.ID:
            raise RuntimeError(f"unexpected core ID 0x{ident:08X}, expected 0x{self.ID:08X} (is the right bitstream programmed?)")

    def _can_connect(self):
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
            return True
        except OSError:
            return False

    def cmd(self, line):
        self.f.write(line + "\n"); self.f.flush()
        resp = self.f.readline()
        if not resp:
            raise RuntimeError("System Console server closed the connection")
        rc, _, res = resp.rstrip("\n").partition(" ")
        if rc != "0":
            raise RuntimeError(f"tcl error for '{line}': {res}")
        return res

    def rd(self, word_addr, n=1):
        res = self.cmd(f"master_read_32 $m 0x{4 * word_addr:X} {n}")
        return [int(x, 16) for x in res.split()]

    def wr(self, word_addr, value):
        self.cmd(f"master_write_32 $m 0x{4 * word_addr:X} 0x{value & 0xFFFFFFFF:X}")

    def wr_pixels(self, data: bytes):
        fd, path = tempfile.mkstemp(suffix=".bin"); os.write(fd, data); os.close(fd)
        try:
            self.cmd("master_write_from_file $m {" + path.replace("\\", "/") + "} 0x%X" % WIN)
        finally:
            os.remove(path)

    @staticmethod
    def s32(v):
        return v - (1 << 32) if v & 0x80000000 else v

    def status(self):
        s = self.rd(A_STATUS)[0]
        return dict(ready=s & 1, loading=(s >> 1) & 1, done=(s >> 2) & 1, ev_overflow=(s >> 3) & 1, pll=(s >> 4) & 1, dropped=(s >> 5) & 1)

    def soft_reset(self):
        self.wr(A_CTRL, 2); time.sleep(0.05)

    def run_frame(self, pixels, pfa_sel=0, tau_q14=12288, theta=-(2 ** 31) + 1, timeout=120.0):
        px = np.ascontiguousarray(pixels, dtype=np.uint8)
        assert px.shape == (self.h, self.w), px.shape
        self.wr(A_CFG, pfa_sel); self.wr(A_TAU, tau_q14); self.wr(A_THETA, theta)
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
        nev, nacc, nres, pix, post = (self.rd(a)[0] for a in (A_NEV, A_NACC, A_NRES, A_PIX, A_POST))
        words = self.rd(A_RES, 2 * nres) if nres else []
        lo = np.array(words[0::2], dtype=np.int64); hi = np.array(words[1::2], dtype=np.int64)
        hi = np.where(hi >= 1 << 31, hi - (1 << 32), hi)
        return dict(n_events=nev, n_accepted=nacc, n_res=nres, pix_issued=pix, post_cycles=post, status=st,
                    j=(lo >> 10) & 0x3FF, i=lo & 0x3FF, accept=(lo >> 29) & 1, logit=hi,
                    t_upload_s=t2 - t1, t_total_s=time.time() - t0)

    def close(self):
        try:
            self.sock.close()
        finally:
            if self.proc:
                self.proc.terminate()
