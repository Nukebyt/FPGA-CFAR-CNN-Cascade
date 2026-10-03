# JTAG-fed cascade (Paper 2 board flow for many full frames)

The 128x128 demo (`../cascade_de10_top.v`) streams one crop from ROM. This design lets the PC push **any 800x800 image**
into the same cascade over the USB-Blaster JTAG cable and read the candidate events + CNN logits back -- no HPS, no Linux,
no SD card, no frame buffer.

```
PC  python (board.py) --TCP--> System Console (jtag_server.tcl) --JTAG--> altera_jtag_avalon_master (Qsys jtag_sys)
                                                                   --Avalon-MM--> cascade_jtag_core -> cascade_top_2clk
```

* `cascade_jtag_core.v` -- Avalon slave (registers, pixel window, result RAM) + clock gating + `cascade_top_2clk`. Register map in its header.
* `gclk_gate.v` -- clock-control block (`altclkctrl`, glitch-free, registered enable). **Why:** the Weibull line buffer needs a gap-free
  pixel stream (`KNOWN_ISSUE_GAP_INTOLERANCE.md`) and a 5.1 Mbit full frame does not fit on-chip next to the cascade. While a frame
  loads, the stream-domain clock only ticks on cycles that carry a pixel, so the core sees a perfect stream however slowly (or
  irregularly) the host writes. `-DSIM_GATE` selects a behavioural model for simulation.
* `cascade_jtag_top.v` -- DE10-Standard top (PLL, Qsys JTAG master, LEDs/HEX). Project: `_quartus/cascade_jtag/`
  (`make_jtag_sys.tcl` -> `qsys-script` -> `qsys-generate`, then `quartus_sh --flow compile cascade_jtag_top`).
* `tb/cascade_jtag_tb.v`, `build_jtag_tb.sh` -- simulation: random-gap pixel feed, readback through the Avalon port; same logs as the other
  cascade TBs so `tb/check_cascade.py` verifies the pooled store, events and every logit independently. Second frame back-to-back checks reuse.
* Host: `_comparison/cnn/sweep250/board/` (`board.py`, `jtag_server.tcl`, `board_sweep.py`).

Programming: `quartus_pgm -m jtag -c "DE-SoC [USB-1]" -o "p;output_files/cascade_jtag_top.sof@2"`.

Measured on the board: a full 800x800 frame uploads in ~0.25 s over JTAG; the CNN pass costs ~0.2 ms per candidate event
(`POST_CYC` register, 20 ns units, includes the pipeline flush).

Capacity: up to 8192 candidate events per frame (`EV_AW=13`); `ev_overflow` is flagged if exceeded. With the 1e-3 plane at tau 0.75,
17 of 250 test images exceed the original 1024-event capacity (worst 4153) -- see the study doc.
