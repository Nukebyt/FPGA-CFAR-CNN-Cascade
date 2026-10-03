function [detection_map, threshold_map_log, shape_map, c1_map, c2_map, stats] = ...
    WeibullCFAR_FixedPoint(I, sli, guard, PfaIndex, lut_dir)
%WEIBULLCFAR_FIXEDPOINT  Step 6 — fixed-point, LUT-driven Weibull CFAR.
%
%   Bit-accurate MATLAB reference model for the streaming FPGA datapath.
%   Uses ONLY the three LUTs produced by generate_LUTs.m (no log/sqrt/exp/
%   fractional-power calls) and reciprocal-multiply-by-constant in place
%   of every division whose divisor is known at compile time (N, N-1, and
%   the c2-LUT address range are all fixed once sli/guard are chosen).
%
%   The one division NOT eliminated is K(Pfa)/C, since C is a per-pixel
%   runtime value from the C-LUT, not a compile-time constant. See notes
%   at the end of this file for how to remove it if needed.
%
%   INPUTS
%     I        : HxW SAR intensity image, values in [0,255]
%     sli      : sliding window size (odd, >=3) -- must match the sli_lut
%                used when generate_LUTs.m built c_lut.mat
%     guard    : guard region size (odd, >=1, <sli) -- must match guard_lut
%     PfaIndex : 1..4, selects Pfa_options(PfaIndex) from pfa_constant_lut
%                (i.e. Pfa in {1e-3, 1e-4, 1e-5, 1e-6})
%     lut_dir  : folder containing log_amp_lut.mat, c_lut.mat,
%                pfa_constant_lut.mat (as written by generate_LUTs.m)
%
%   OUTPUTS
%     detection_map    : HxW logical, x_code > T_log_code
%     threshold_map_log : HxW log-domain threshold, DEQUANTIZED (double)
%                          for inspection/plotting only
%     shape_map        : HxW Weibull shape parameter C, DEQUANTIZED
%     c1_map, c2_map    : HxW log-cumulants, DEQUANTIZED, for diagnostics
%     stats             : struct, see below
%
%   stats fields:
%     NumDetections, Pfa_used, MeanShapeParameter,
%     FractionC2ClampedLow, FractionC2ClampedHigh  (fraction of pixels
%       whose c2 fell outside the C-LUT's built c2 range and was
%       saturated to an end entry -- watch this; high values here mean
%       the LUT's c2 range from generate_LUTs.m no longer covers this
%       run's data and the LUT should be regenerated)
%
%   See also: WeibullCFAR_Floating, generate_LUTs

%% ---- Validate + load LUTs ------------------------------------------------
if mod(sli,2)==0 || sli<3,  error('sli must be odd and >=3.'); end
if mod(guard,2)==0 || guard<1 || guard>=sli, error('guard must be odd, >=1, <sli.'); end
if ~ismember(PfaIndex, 1:4), error('PfaIndex must be 1..4.'); end

L = load(fullfile(lut_dir,'log_amp_lut.mat'));   % log_amp_lut_fixed_int, LOGAMP_FRAC_BITS
Cl = load(fullfile(lut_dir,'c_lut.mat'));         % c_lut_fixed_int, c2_lut_min/max, CLUT_FRAC_BITS
Kl = load(fullfile(lut_dir,'pfa_constant_lut.mat')); % K_fixed_int, KLUT_FRAC_BITS, Pfa_options

LOGAMP_FRAC = L.LOGAMP_FRAC_BITS;      % 13
CLUT_FRAC   = Cl.CLUT_FRAC_BITS;       % 12
KLUT_FRAC   = Kl.KLUT_FRAC_BITS;       % 12
N_C_ENTRIES = numel(Cl.c_lut_fixed_int);

I = double(I);
[h, w] = size(I);
N = sli^2 - guard^2;   % reference-cell count, compile-time constant given sli/guard

%% ---- Stage 1: log-amplitude, via LUT (no log/sqrt at runtime) -----------
% Direct table lookup: intensity 0..255 -> Q2.13 code. This is the ONLY
% place the raw image value is used; everything downstream works on
% x_code (integer, Q2.13-scaled).
idx = round(I) + 1;                      % I in [0,255] -> 1-based LUT index
idx = min(max(idx, 1), 256);             % defensive clamp, e.g. saturated pixels
x_code = L.log_amp_lut_fixed_int(idx);   % HxW, integer-valued double

%% ---- Stage 2: local (bounded) box sums, x and x^2 ------------------------
% See file header: separable movsum, NOT a full-image cumsum. Magnitude of
% every intermediate value here is bounded by (window size) x (max |term|),
% independent of image size -- this is what a real line-buffer/shift-
% register accumulator in RTL will look like.
sum_x_sli  = local_box_sum_fixed(x_code, sli);
sum_x_grd  = local_box_sum_fixed(x_code, guard);
sum_x_ref  = sum_x_sli - sum_x_grd;             % Q2.13-scaled, signed

% x^2, requantized back down to the SAME 13 fractional bits as x before
% summing (a real multiplier would produce a much wider product; hardware
% truncates/rounds it back to a fixed output width before accumulating --
% modelled explicitly here rather than silently kept at full precision).
x2_code_full = x_code .^ 2;                     % Q4.26-scale (unsigned, x^2>=0)
x2_code      = round(x2_code_full / 2^LOGAMP_FRAC);   % requantized to Q?.13

sum_x2_sli = local_box_sum_fixed(x2_code, sli);
sum_x2_grd = local_box_sum_fixed(x2_code, guard);
sum_x2_ref = sum_x2_sli - sum_x2_grd;           % Q?.13-scaled, unsigned

%% ---- Stage 3: c1, c2 via reciprocal-multiply (N, N-1 are compile-time) --
% c1 = sum_x_ref / N  ==>  c1_code = round(sum_x_ref_code * invN / 2^M)
% This is exactly how the RTL should do it too: N is fixed once sli/guard
% are chosen, so 1/N is a SINGLE precomputed constant, not a runtime
% divider. M=16 gives ~1e-5 relative precision on the reciprocal, far
% below the LUT's own resolution -- more than sufficient.
M = 24;
invN   = round(2^M / N);
invNm1 = round(2^M / (N - 1));

c1_code = round(sum_x_ref .* invN / 2^M);       % Q2.13-scaled, signed

c1_sq_full = c1_code .^ 2;                       % Q4.26-scale
c1_sq      = round(c1_sq_full / 2^LOGAMP_FRAC);  % requantized to Q?.13 (matches sum_x2_ref's scale)

c2_num  = sum_x2_ref - N .* c1_sq;               % Q?.13-scaled, signed (variance numerator)
c2_code = round(c2_num .* invNm1 / 2^M);         % Q?.13-scaled c2

% NOTE: WeibullCFAR_Floating.m applies an epsilon floor (c2 = max(c2,1e-6))
% to guard against a zero/negative variance. That floor (1e-6) is roughly
% 100x below Q2.13's own resolution (2^-13 ~= 1.22e-4) and would just
% quantize to code 0 here -- it is NOT separately needed in fixed point,
% because the C-LUT's own c2_lut_min clamp already maps any very-small or
% non-positive c2 to its first entry, which was built as C_MAX (see
% generate_LUTs.m Step 4a) -- exactly the same protective behaviour,
% for free, from the LUT's existing range clamp.

%% ---- Stage 4: c2 -> C-LUT address -> C, via LUT (no sqrt/division) ------
c2_actual = c2_code / 2^LOGAMP_FRAC;             % dequantize for address calc
c2_lut_min = Cl.c2_lut_min;  c2_lut_max = Cl.c2_lut_max;

% Address = round((c2-min)/(max-min) * (N_entries-1)). min/max/N_entries
% are ALL compile-time constants (fixed once c_lut.mat is generated), so
% this too reduces to reciprocal-multiply-by-constant + shift in RTL --
% no runtime divider needed here either.
addr_frac = (c2_actual - c2_lut_min) / (c2_lut_max - c2_lut_min) * (N_C_ENTRIES - 1);
addr = round(addr_frac);

clamped_low  = addr < 0;
clamped_high = addr > (N_C_ENTRIES - 1);
addr = min(max(addr, 0), N_C_ENTRIES - 1);

C_code = Cl.c_lut_fixed_int(addr + 1);           % direct LUT read, Q4.12 unsigned
C_actual = C_code / 2^CLUT_FRAC;

%% ---- Stage 5+6: K/C via combined LUT — NO runtime division -------------
KCl = load(fullfile(lut_dir,'kc_lut.mat'));
KCLUT_FRAC = KCl.KCLUT_FRAC_BITS;
N_C_ENTRIES_kc = KCl.N_C_LUT_ENTRIES;   % sanity: should equal N_C_ENTRIES above

flat_addr = (PfaIndex-1)*N_C_ENTRIES_kc + addr;   % addr already computed in Stage 4
KC_code   = KCl.KC_flat(flat_addr + 1);
KC_actual = KC_code / 2^KCLUT_FRAC;
Pfa_used  = KCl.Pfa_options(PfaIndex);

c1_actual = c1_code / 2^LOGAMP_FRAC;
T_log_actual = c1_actual + KC_actual;

x_actual = x_code / 2^LOGAMP_FRAC;

%% ---- Stage 7: decision -----------------------------------------------
detection_map = x_actual > T_log_actual;

%% ---- Outputs / stats ------------------------------------------------
threshold_map_log = T_log_actual;
shape_map = C_actual;
c1_map = c1_actual;
c2_map = c2_actual;

stats = struct();
stats.NumDetections = sum(detection_map(:));
stats.Pfa_used = Pfa_used;
stats.MeanShapeParameter = mean(shape_map(:));
stats.FractionC2ClampedLow  = mean(clamped_low(:));
stats.FractionC2ClampedHigh = mean(clamped_high(:));

end % === end WeibullCFAR_FixedPoint =======================================


function S = local_box_sum_fixed(x_code, k)
%LOCAL_BOX_SUM_FIXED  k x k box sum via a SEPARABLE, BOUNDED moving sum:
%   row-pass (horizontal window, magnitude bounded by k*max|term|) then
%   column-pass on that result (vertical window, bounded by k*that bound).
%   This intentionally mirrors a real line-buffer/shift-register
%   architecture (a row of k-tap running sums, accumulated down k rows) --
%   at no point does any intermediate value depend on image size, unlike
%   a full-image cumsum/integral-image, which is why this replaces
%   WeibullCFAR_Floating.m's local_box_sum for the fixed-point model.
    tk = (k - 1) / 2;
    xp = padarray(x_code, [tk, tk], 'symmetric', 'both');
    row_pass = movsum(xp, k, 2, 'Endpoints', 'discard');   % horizontal (columns)
    S        = movsum(row_pass, k, 1, 'Endpoints', 'discard'); % vertical (rows)
end