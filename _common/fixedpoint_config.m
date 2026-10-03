function C = fixedpoint_config(sli, guard)
%FIXEDPOINT_CONFIG  Every Q-format in the fixed-point datapath, in one place.
%
%   C = FIXEDPOINT_CONFIG(sli, guard)
%
%   Each format below traces to a measured or analytically-derived number
%   printed by _comparison/analyse_fixedpoint_ranges.m. Nothing here is
%   guessed; re-run that script if the window geometry or the input bit depth
%   changes, and update this file from its output.
%
%   ---------------------------------------------------------------------
%   THE CENTRED LOG-AMPLITUDE, AND WHY IT BUYS A BIT
%   ---------------------------------------------------------------------
%   The hardware log LUT stores xc = log(sqrt(I+0.5)) - X0, not the raw
%   log-amplitude. Subtracting the constant X0 is free (it is baked into the
%   table at generation time) and it does two things:
%
%     1. It removes the mean-induced cancellation that would otherwise
%        destroy c3 in fixed point (BUG_LOG H1). This is the reason it
%        exists.
%     2. It makes the range symmetric about zero. Raw x spans
%        [-0.3466, +2.7716] and needs 2 integer bits; centred xc spans
%        [-1.5591, +1.5591] and needs only 1. The existing Weibull RTL,
%        which does not centre, uses Q2.13 for this signal; the shared
%        front end here uses Q1.14 and is strictly more precise at the same
%        16-bit width.
%
%   ---------------------------------------------------------------------
%   ACCUMULATOR WIDTHS
%   ---------------------------------------------------------------------
%   Sized from the analytical worst case N*max|term|, NOT from the empirical
%   distribution: an accumulator that overflows on a pathological window is a
%   hardware failure, not a statistical outlier, so the bound has to be the
%   true one. At sli=17/guard=13 (N=120) that is 23-24 bits; widths are
%   recomputed here from (sli,guard) so changing the geometry cannot silently
%   under-size them.
%
%   ---------------------------------------------------------------------
%   WHAT IS *NOT* HERE
%   ---------------------------------------------------------------------
%   LUT address ranges. Those are a per-detector decision made in each
%   generate_LUTs_*.m, because the right range is set by the detector's
%   SUPPORT CONDITION (FINDINGS F4), not by this file's signal ranges. That
%   distinction matters: the empirical range of the Burr address variable
%   s = c3/c2^1.5 runs to +-75 on real data, but every value outside
%   (-1.139443, 2) is out of support and is rejected before the table is ever
%   addressed -- so the table only has to span the support band, and sizing it
%   from the empirical range would waste ~97% of the entries.
%
%   OUTPUT (struct C), each format given as a .<name> substruct with fields
%     int, frac, total, signed, scale (= 2^frac), min, max, res

if nargin < 1, sli   = 17; end
if nargin < 2, guard = 13; end

C = struct();
C.sli   = sli;
C.guard = guard;
C.N     = sli^2 - guard^2;

%% ---- The centring constant ---------------------------------------------
% Midpoint of the 8-bit log-amplitude range. Chosen once and frozen: it is
% baked into log_amp_lut.hex, so changing it invalidates that table.
C.X0 = 1.2125;

x_lo = log(sqrt(0   + 0.5));
x_hi = log(sqrt(255 + 0.5));
C.x_min = x_lo;
C.x_max = x_hi;
C.xc_max = max(abs([x_lo, x_hi] - C.X0));      % 1.5591

%% ---- Stage 1: log-amplitude LUT output ---------------------------------
% |xc| <= 1.5591 -> 1 integer bit + sign. 14 fractional bits at 16 total.
C.xc = fmt(1, 14, true);

%% ---- Stage 2: powers, requantised before accumulation ------------------
% A real multiplier produces a double-width product; hardware truncates it
% back to a fixed width before accumulating. Modelled explicitly rather than
% silently kept at full precision.
%
% *** 18 FRACTIONAL BITS, NOT 13 -- SIZED FROM A MEASURED FAILURE ***
% The first version used 13, matching the existing Weibull RTL. That is
% sufficient for Weibull (which passed the gate at 0.0033%) but NOT for the
% three-parameter detectors, and the Phase 3 gate caught it:
%
%   Generalized Gamma addresses its shape ROM by log2|s| = log2|c3| -
%   1.5*log2(c2). A LOG address converts a RELATIVE error in c2/c3 into an
%   ABSOLUTE address error. With 13 fractional bits and typical c2 ~ c3 ~
%   0.08, the relative quantisation error is ~7.5e-4, giving a median
%   log2|s| error of 2.6e-3 -- about 2.5x the shape table's own address step.
%   The address error, not the table, dominated; and near k = KMin the map
%   from address to delta is steep enough that it pushed windows across the
%   k clamp boundary, producing threshold errors up to 0.09 and a 0.088%
%   detection mismatch (gate is 0.01%).
%
% *** AND 24, NOT 18 -- THE CANCELLATION IS WORSE THAN THE QUANTISATION ***
% Widening to 18 helped (0.088% -> 0.065% mismatch) but did not close the
% gate, because the dominant error on the worst windows is CANCELLATION, not
% quantisation. The moments are formed as
%     m2 = S2/N - m1^2
%     m3 = S3/N - 3*m1*(S2/N) + 2*m1^3
% and on a near-uniform window sitting well away from X0 -- bright, flat sea,
% which is most of a SAR scene -- every term is of order 1 while the result is
% of order 1e-4. Measured worst case: c2 = 6.8e-4 and c3 = 5.0e-5 from terms
% all close to 1.0, i.e. four decimal digits of cancellation.
%
% Centring by the fixed constant X0 (see below) removes the cancellation only
% for windows whose mean is near X0; it cannot help a window whose own mean is
% far from it, and a single-pass streaming architecture cannot re-centre per
% window without buffering it. The only single-pass fix is to carry enough
% fractional bits that four digits of cancellation still leaves the result
% accurate, which is what 24 does.
%
% Cost: S2/S3 grow to ~33-34 bits. These are ACCUMULATORS -- adder chains and
% registers, not multipliers -- and DSP count, not adder width, is the binding
% resource on Cyclone V (73% DSP vs 51% ALM at SLI=18 for the existing
% two-moment Weibull build). Cheap in the currency that is actually scarce.
%
% Weibull does not need any of this: it addresses on c2 LINEARLY, so it sees
% only the absolute error, and it passed the gate at 13 bits. The precision is
% carried for it anyway because the front end is shared.
C.xc2 = fmt(2, 24, false);      % xc^2 in [0, 2.4308]
C.xc3 = fmt(2, 24, true);       % xc^3 in [-3.7899, +3.7899]

%% ---- Stage 3: windowed accumulators ------------------------------------
% Analytical worst case: every cell at the extreme. Never expected in
% practice; must not overflow anyway.
C.S1 = acc_fmt(C.N * C.xc_max,     C.xc.frac,  true);
C.S2 = acc_fmt(C.N * C.xc_max^2,   C.xc2.frac, false);
C.S3 = acc_fmt(C.N * C.xc_max^3,   C.xc3.frac, true);

%% ---- Stage 4: log-cumulants --------------------------------------------
% c1 is reported in the UNCENTRED domain (X0 added back), so it spans the
% raw x range, not the centred one.
% c2 and c3 carry the same widened precision as the powers they come from --
% see the note on C.xc2/C.xc3 above. Their fractional widths set the relative
% accuracy of the log-domain shape address, which is what the two
% three-parameter detectors are sensitive to.
% *** SIZED ANALYTICALLY, NOT FROM THE EMPIRICAL MAX ***
% The first version used 1 integer bit for both c2 and c3, on the grounds that
% the measured maxima over 80 SSDD images were 1.906 and 1.64 -- comfortably
% under 2. That is exactly the mistake fixedpoint_config's own header warns
% about for accumulators, made one stage later: on a 50-image run that had not
% been sampled during sizing, the gate reported 19 overflows (BUG_LOG D11).
%
% The true bounds follow from the centred range |xc| <= 1.5591:
%   c2 = E[(xc-m1)^2] <= (2*1.5591/2)^2 * N/(N-1)  ~ 2.45   -> 2 integer bits
%   |c3| = |E[(xc-m1)^3]| <= max|xc-m1| * E[(xc-m1)^2]
%                          <= 3.118 * 2.45 ~ 7.6            -> 3 integer bits
% Both are hard bounds, not percentiles, so no dataset can exceed them.
C.c1 = fmt(2, 13, true);        % [-0.35, 2.78]; addressed linearly, 13 is ample
C.c2 = fmt(2, 24, false);       % hard bound 2.45
C.c3 = fmt(3, 24, true);        % hard bound 7.6

% *** m1 IS THE ONE THAT ACTUALLY MATTERED ***
% m1 = S1/N is an intermediate, not an output, and the first version carried
% it at the same 14 fractional bits as xc. That was the real precision
% bottleneck, and widening xc2/xc3 from 13 to 18 to 24 changed the gate result
% by literally nothing (0.06474% at every width) because m1 was capping it:
%
%     m2 = S2/N - m1^2        m3 = S3/N - 3*m1*(S2/N) + 2*m1^3
%
% For a bright window m1 ~ 1, so an absolute error e in m1 puts ~2*m1*e into
% m1^2 and straight into c2 and c3. At 14 fractional bits e ~ 3e-5, giving the
% ~4e-5 absolute error measured in both c2 and c3 -- about 700x the c3
% quantisation step, i.e. entirely dominated by this one truncation.
%
% m1 can legitimately carry far more: it is the mean of N terms, so the
% rounding noise in S1 averages down by ~sqrt(N) and S1/N is meaningful well
% past xc's own resolution. Carrying it at 24 bits costs one wider register
% and one wider multiply, and removes the bottleneck.
%
% LESSON (BUG_LOG D8): widen the signal the error analysis points at, not the
% one that looks widest in the block diagram. Three rounds of widening the
% wrong signal produced bit-identical results before the front ends were
% compared directly.
C.m1 = fmt(1, 24, true);

%% ---- Reciprocal-multiply constants (compile-time, given sli/guard) -----
% Division by N and N-1 and N-2 is never done at runtime: N is fixed once the
% geometry is chosen, so each reciprocal is a single stored constant. M=24
% gives ~6e-8 relative error, far below every downstream LUT's resolution.
% *** M CHOSEN BY SWEEP, NOT BY ARGUMENT -- FPGA checklist section 1.4 ***
% An earlier version set M = 24 with an analytic justification. The checklist
% is explicit that this is not good enough ("sweep M against exact-fraction
% arithmetic before picking it -- don't guess"), and the sweep
% (_comparison/audit_recip_precision.m, N = 120) shows why:
%
%     M     wrong roundings     max err (LSB)
%     16        92.0%              6.24
%     20        20.0%              0.390
%     24         1.6%              0.0244      <- the guessed value
%     28         0.8%              0.00152     <- 16x better, ~free
%
% wrong% is also NOT monotonic in M (26 beats 28 on that metric), which is
% exactly the kind of behaviour an analytic argument does not predict. Max
% error is what propagates into c2/c3 through m1^2 and m1^3, so M = 28 is
% adopted. The cost is four extra bits on one constant multiplier, against
% BUG_LOG D8's demonstration that m1 precision dominates both cumulants.
%
% Measured bias is ~0 at every M tested, which was the real risk: a small but
% systematically-signed reciprocal error would accumulate into a threshold
% offset rather than averaging out.
C.RECIP_M = 28;

% *** THESE ARE round(), NOT floor() -- AND THE DIFFERENCE IS REAL ***
% Verilog's `/` on integers TRUNCATES, so an RTL expression `(1<<M)/N` does
% NOT reproduce these constants. Measured divergences at M = 24:
%     sli=21/guard=15 : 1/(N-1) -> round 78034  vs floor 78033
%     sli=15/guard=11 : 1/(N-1) -> round 162886 vs floor 162885
%     sli=31/guard=21 : 1/N     -> round 32264  vs floor 32263
% i.e. 3 of 4 geometries tested. Per checklist section 1.11, the RTL must
% HARDCODE the values printed by _comparison/audit_recip_precision.m rather
% than compute them, or the golden model and the hardware quietly use
% different constants -- the failure mode where "the RTL becomes more accurate
% than the thing verifying it."
C.invN    = round(2^C.RECIP_M / C.N);
C.invNm1  = round(2^C.RECIP_M / (C.N - 1));
% k-statistic scale factors (cfar_front_end uses unbiased k2, k3):
%   k2 = N/(N-1) * m2      k3 = N^2/((N-1)(N-2)) * m3
C.k2_scale = C.N / (C.N - 1);
C.k3_scale = C.N^2 / ((C.N - 1) * (C.N - 2));

%% ---- Stage 5: shape / back-end tables ----------------------------------
% delta = T_log - c1. Measured max across all five detectors and the
% {1e-3..1e-6} Pfa set is 30.5 (Burr XII at Pfa=1e-6), so 5 integer bits.
% Kept common across detectors so one comparator width serves all of them.
C.delta = fmt(5, 10, true);

% sqrt(c2): shared by Weibull, Lognormal, Generalized Gamma and Burr XII
% (FINDINGS F2, F10). c2 < 2 => sqrt(c2) < 1.415, so 1 integer bit.
C.sqrt_c2 = fmt(1, 14, false);

% T_log itself = c1 + delta.
C.T_log = fmt(6, 10, true);

%% ---- Default table sizes ------------------------------------------------
% 4096 entries = 12-bit address. At 16 bits/entry that is 64 Kbit per table,
% ~1.1% of a Cyclone V's 5.6 Mbit budget. Each generate_LUTs_*.m prints a
% size-vs-error sweep so this can be revisited per detector rather than
% assumed.
C.N_SHAPE_ENTRIES = 4096;
C.N_LOGAMP_ENTRIES = 256;      % fixed: one entry per 8-bit input value
end


%% =======================================================================
function f = fmt(intBits, fracBits, isSigned)
    f = struct();
    f.int    = intBits;
    f.frac   = fracBits;
    f.signed = isSigned;
    f.total  = intBits + fracBits + double(isSigned);
    f.scale  = 2^fracBits;
    f.res    = 2^-fracBits;
    if isSigned
        f.min = -2^intBits;
        f.max =  2^intBits - f.res;
    else
        f.min = 0;
        f.max =  2^intBits - f.res;
    end
end

function f = acc_fmt(maxAbs, fracBits, isSigned)
%ACC_FMT  Accumulator format sized to hold maxAbs exactly.
    intBits = max(0, ceil(log2(max(maxAbs, eps))));
    f = fmt(intBits, fracBits, isSigned);
    f.boundAbs = maxAbs;
end
