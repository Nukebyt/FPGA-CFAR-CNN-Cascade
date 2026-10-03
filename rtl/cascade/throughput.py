# -*- coding: utf-8 -*-
"""Frame-rate model from MEASURED RTL cycle counts (cnn_core_tb) -- validated against the simulated
two-clock frame (128x128, SMALL, 24 events: model 154.4k clk cycles vs measured 154,491)."""
import numpy as np

FETCH = 1056 + 12            # patch_fetch (32 rows x (1+32) + flush) + controller handshakes, CNN-clock cycles
STREAM_CLK = 50e6
# measured compute cycles per patch (end of load -> done)
CORE = {"SMALL": {"serial": 40135, "quad": 10443},
        "DEEP":  {"serial": 71012, "quad": 19658},
        "XL":    {"serial": 171904 + 0, "quad": 48174}}   # XL serial = issue-cycle estimate (not simulated)
W = H = 800
EV = 274                      # gated events per 800x800 test image (hardware-exact test split mean)


def frame_s(core_cyc, f_cnn):
    t_stream = (W * H + 2000) / STREAM_CLK
    return t_stream + EV * (core_cyc + FETCH) / f_cnn


print(f"{'model':6s} {'serial@50':>10s} {'quad@50':>10s} {'quad@100':>10s}   (frames/s at {W}x{H}, {EV} events/frame)")
for m, c in CORE.items():
    r = [1 / frame_s(c["serial"], 50e6), 1 / frame_s(c["quad"], 50e6), 1 / frame_s(c["quad"], 100e6)]
    print(f"{m:6s} {r[0]:10.1f} {r[1]:10.1f} {r[2]:10.1f}   speedup quad@100 vs serial@50: {r[2]/r[0]:.1f}x")
print("\nvalidation: 128x128 SMALL quad, 24 events, 100 MHz CNN clock ->",
      round(((128*128) + 24 * (10443 + FETCH) / 2.0)), "clk(50MHz) cycles  (measured 154,491)")
