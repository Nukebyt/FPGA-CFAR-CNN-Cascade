function [detection_map, threshold_map, shape_map, stats, logdomain] = ...
    WeibullCFAR_Floating(I, sli, guard, Pfa, varargin)
%WEIBULLCFAR_FLOATING  Classical single-pass windowed Weibull CFAR detector.
%
%   [detection_map, threshold_map, shape_map, stats] = ...
%       WeibullCFAR_Floating(I, sli, guard, Pfa)
%
%   [..., logdomain] = WeibullCFAR_Floating(...)
%
%   This is the "clean" single-pass floating-point reference model that is
%   the algorithmic starting point for the streaming FPGA implementation
%   (Cyclone IV / DE2, Quartus Prime). It is a DELIBERATE rewrite of the
%   original two-pass model (Main_CFAR_Weibull.m), not an incremental edit
%   of it. Per the project brief, this function does NOT:
%
%       - build a rough Otsu target mask
%       - run bwareaopen / morphological dilation
%       - exclude suspected-target pixels from the clutter statistics
%       - fall back to whole-image ("global") statistics for dirty windows
%
%   Every reference-window statistic is computed strictly from the local
%   sli x sli window around the current Cell Under Test (CUT), with the
%   guard cells excluded. That is the only thing the streaming hardware
%   will ever be able to do (a single sliding window, no whole-image
%   passes, no exclusion logic), so the floating-point model is written to
%   match that constraint from the start.
%
%   ---------------------------------------------------------------------
%   STEP 1 — the classical amplitude/intensity-domain detector
%   ---------------------------------------------------------------------
%   Reference-cell geometry: the reference window is the sli x sli
%   neighbourhood of the CUT, minus a concentric guard x guard hole in the
%   middle (the CUT sits at the centre of that hole):
%
%       N = sli^2 - guard^2
%
%   Clutter model (Method of Log-Cumulants, matches wblmolc.m):
%       I_amp = sqrt(I + 0.5)                    (0.5 LSB dequant. offset)
%       x     = log(I_amp)
%       c1    = mean(x over reference cells)
%       c2    = var(x over reference cells)       (SAMPLE variance, N-1 —
%                                                    see note below)
%       C_raw = sqrt(psi(1,1) / c2)              = sqrt((pi^2/6) / c2)
%       C     = clamp(C_raw, C_MIN, C_MAX)
%       B     = exp(c1 + gamma / C)
%       T_amp = B * (-log(Pfa))^(1/C)
%       T_int = T_amp^2                           (intensity-domain thresh)
%
%   IMPORTANT DEVIATION FROM wblmolc.m / the original two-pass model:
%   wblmolc.m (and the original Main_CFAR_Weibull*.m scripts) compute B
%   from the *unclamped* C_raw, and only clamp C for the exponent term of
%   the threshold itself:
%       [B, C] = wblmolc(c1, c2);   % B uses C_raw internally
%       C = min(max(C, C_MIN), C_MAX);
%       thres_amp = B * (-log(Pfa))^(1/C);   % now uses clamped C
%   In the streaming FPGA architecture the shape parameter is produced by
%   a LUT (Step 4) that ALREADY OUTPUTS THE CLAMPED VALUE — the hardware
%   never has access to C_raw. So B must be computed from the SAME
%   (clamped) C that the exponent term uses, or the amplitude-domain and
%   log-domain formulations (Step 2) would not be equivalent whenever a
%   window happens to land past a clamp boundary. This function therefore
%   uses the clamped C for both B and the exponent, everywhere. C_raw is
%   still computed and reported (stats.ShapeParameterRawMap /
%   stats.FractionAtCMin / stats.FractionAtCMax) purely as a diagnostic,
%   e.g. to see how often the clamp is active and to help size the C-LUT
%   in generate_LUTs.m.
%
%   ---------------------------------------------------------------------
%   STEP 2 — the log-domain (FPGA-oriented) decision, verified equivalent
%   ---------------------------------------------------------------------
%   Starting from B = exp(c1 + gamma/C) and T_amp = B*(-log(Pfa))^(1/C):
%
%       log(T_amp) = c1 + gamma/C + log(-log(Pfa))/C
%                  = c1 + (gamma + log(-log(Pfa))) / C
%                  = c1 + K(Pfa) / C                      , K(Pfa) = gamma + log(-log(Pfa))
%
%   Define T_log = c1 + K(Pfa)/C and x_cut = log(sqrt(I_cut + 0.5)). Then
%   x_cut > T_log  <=>  I_amp_cut > T_amp  <=>  I_cut > T_amp^2 = T_int,
%   i.e. it is exactly the same decision, just without exp(), sqrt() or a
%   fractional power at decision time (only a single reciprocal of C,
%   which becomes the C-LUT's job in hardware, and a table lookup for
%   K(Pfa), Step 5). This is what removes exp()/pow() from the streaming
%   critical path. This function always computes BOTH decisions and
%   reports their agreement in the `logdomain` output, so the equivalence
%   is checked on every call rather than trusted blindly.
%
%   ---------------------------------------------------------------------
%   INPUTS
%     I     : 2-D SAR intensity image (numeric, any positive range; cast
%             to double internally).
%     sli   : sliding window size (must be ODD, >= 3). e.g. 15 or 21
%     guard : guard region size  (must be ODD, >= 1, < sli). e.g. 5 or 7
%     Pfa   : desired probability of false alarm, 0 < Pfa < 1.
%
%   NAME-VALUE OPTIONS
%     'CMin'    : lower clamp for Weibull shape parameter C (default 0.8)
%     'CMax'    : upper clamp for Weibull shape parameter C (default 8.0)
%     'Verbose' : true/false, print progress/debug info    (default false)
%
%   OUTPUTS
%     detection_map : logical HxW map, true where the amplitude/intensity
%                     -domain decision fires (this is the "canonical"
%                     output — equivalent to the original two-pass model's
%                     decision rule, minus the exclusion/fallback logic).
%     threshold_map  : HxW intensity-domain threshold T_int used above.
%     shape_map      : HxW CLAMPED Weibull shape parameter C.
%     stats          : struct of summary statistics, see below.
%     logdomain      : struct with the Step 2 log-domain decision and its
%                       agreement with detection_map, see below.
%
%   stats fields:
%     NumReferenceCells, NumDetections, MeanThreshold,
%     MinShapeParameter, MeanShapeParameter, MaxShapeParameter,
%     FractionAtCMin, FractionAtCMax     (fraction of pixels whose RAW,
%                                          unclamped C hit each clamp)
%     C1Map, C2Map, ShapeParameterRawMap (full HxW maps, useful for
%                                          generate_LUTs.m's empirical
%                                          c2-range analysis; NOT reduced
%                                          to scalars so callers can reuse
%                                          them directly)
%
%   logdomain fields:
%     ThresholdMap    : HxW log-domain threshold T_log
%     DetectionMap    : HxW logical, the log-domain decision x > T_log
%     NumMismatch     : number of pixels where DetectionMap ~= detection_map
%     PercentMismatch : 100 * NumMismatch / (h*w)
%
%   See also: wblmolc, Main_CFAR_Weibull (original two-pass reference model)

%% ---- Parse optional arguments -----------------------------------------
p = inputParser;
addParameter(p, 'CMin',    0.8,   @(v) isnumeric(v) && isscalar(v) && v > 0);
addParameter(p, 'CMax',    8.0,   @(v) isnumeric(v) && isscalar(v) && v > 0);
addParameter(p, 'Verbose', false, @(v) islogical(v) && isscalar(v));
parse(p, varargin{:});
C_MIN   = p.Results.CMin;
C_MAX   = p.Results.CMax;
verbose = p.Results.Verbose;

if C_MAX <= C_MIN
    error('WeibullCFAR_Floating:BadClamp', 'CMax must be greater than CMin.');
end

%% ---- Validate mandatory arguments --------------------------------------
if mod(sli, 2) == 0 || sli < 3
    error('WeibullCFAR_Floating:BadWindow', 'sli must be odd and >= 3.');
end
if mod(guard, 2) == 0 || guard < 1 || guard >= sli
    error('WeibullCFAR_Floating:BadGuard', 'guard must be odd, >= 1, and < sli.');
end
if Pfa <= 0 || Pfa >= 1
    error('WeibullCFAR_Floating:BadPfa', 'Pfa must satisfy 0 < Pfa < 1.');
end

I = double(I);
[h, w] = size(I);

tsli   = (sli   - 1) / 2;   % half-width of the outer reference window
tguard = (guard - 1) / 2;   % half-width of the guard hole

%% ---- Amplitude / log-amplitude domain ----------------------------------
% Pre-processing is unchanged from the original reference model: a 0.5 LSB
% dequantization offset is added before the sqrt.
I_amp = sqrt(I + 0.5);
x     = log(I_amp);          % log-amplitude image; c1/c2 are computed on x

%% ---- Reference-cell count -----------------------------------------------
N = sli^2 - guard^2;
if N < 2
    error('WeibullCFAR_Floating:TooFewRefCells', ...
        'sli/guard combination leaves fewer than 2 reference cells (N=%d).', N);
end

%% ---- Symmetric padding, sized for the largest (sli) window --------------
% NOTE: 'symmetric' padding matches the original two-pass model and is
% fine for this floating-point reference. The streaming line-buffer
% architecture (Step 6+) cannot literally mirror-pad an infinite stream;
% it will need its own boundary-handling decision (e.g. replicate the
% first/last valid window, or simply not declare detections in the first/
% last tsli rows/cols of a frame). That is out of scope here and is noted
% again in the chat-level summary as a Step 6 item.
x_pad = padarray(x, [tsli, tsli], 'symmetric', 'both');

%% ---- Windowed sums via a summed-area table (integral image) -------------
% The reference window is a fixed sli x sli square minus a concentric
% guard x guard square, both centred on the CUT. Rather than looping over
% every pixel AND every window element (an O(h*w*sli^2) loop, as in the
% two-pass model), a summed-area table gives sum(x) and sum(x.^2) over any
% centred square in O(1) per pixel after an O(h*w) precompute. This is
% also the natural floating-point analogue of the running-sum / line-
% buffer accumulator this block becomes in hardware (window_sum.sv /
% line_buffer.sv): a box sum is exactly "add the new edge, drop the old
% edge" done all at once instead of incrementally.
sum_x_sli  = local_box_sum(x_pad,    tsli, sli);
sum_x2_sli = local_box_sum(x_pad.^2, tsli, sli);
sum_x_grd  = local_box_sum(x_pad,    tsli, guard);
sum_x2_grd = local_box_sum(x_pad.^2, tsli, guard);

sum_x_ref  = sum_x_sli  - sum_x_grd;   % sum(x)    over reference cells only
sum_x2_ref = sum_x2_sli - sum_x2_grd;  % sum(x.^2) over reference cells only

%% ---- Method-of-log-cumulants Weibull parameters (per pixel) -------------
c1 = sum_x_ref / N;                                    % sample mean

% Sample variance (N-1 denominator) — this MATCHES MATLAB's var() default
% ("normalized by N-1"), which is what the original model relies on via
% c2 = var(clutter_amp_log). Preserved here exactly, per the project brief.
c2 = (sum_x2_ref - N .* c1.^2) / (N - 1);
c2 = max(c2, 1e-6);   % guards against a numerically negative/zero variance
                       % from floating-point cancellation on a flat window;
                       % never expected to trigger on real SAR clutter.

PSI11   = pi^2 / 6;              % psi(1,1), trigamma(1) — exact closed form
EULER_GAMMA = 0.5772156649015329; % -psi(1), Euler-Mascheroni constant

C_raw     = sqrt(PSI11 ./ c2);                       % unclamped shape param
shape_map = min(max(C_raw, C_MIN), C_MAX);           % CLAMPED — used below

% B is computed from the CLAMPED shape parameter — see the "IMPORTANT
% DEVIATION" note in the file header for why this differs from wblmolc.m.
B = exp(c1 + EULER_GAMMA ./ shape_map);

%% ---- Amplitude threshold -> intensity-domain threshold -------------------
T_amp = B .* ((-log(Pfa)) .^ (1 ./ shape_map));

% IMPORTANT: I_amp = sqrt(I + 0.5), so I_amp^2 = I + 0.5, NOT I. The decision
% I_amp_cut > T_amp is therefore equivalent to (I_cut + 0.5) > T_amp^2, i.e.
%   I_cut > T_amp^2 - 0.5
% The original two-pass model used "thres_intensity = thres_amp^2" and
% compared raw I against THAT (no -0.5); that convention is carried forward
% unmodified everywhere it is only ever compared against itself, but it is
% NOT exactly the same decision as I_amp_cut > T_amp. That 0.5-intensity-
% unit gap is invisible in a single-domain model, but it does NOT cancel
% against the Step 2 log-domain formula (which reflects the true amplitude
% comparison exactly) -- it shows up as real (non-rounding) mismatches on
% integer-valued image data, wherever I happens to land inside that half-
% LSB band. The -0.5 below removes that gap so this threshold is exactly
% the intensity-domain equivalent of "I_amp_cut > T_amp":
threshold_map = T_amp .^ 2 - 0.5;   % intensity-domain threshold

% Defensive clamp only — not a statistical cap, kept for parity with the
% original model's defensive check. Not expected to trigger in practice.
bad = ~isfinite(threshold_map) | threshold_map <= -0.5;
if any(bad(:))
    threshold_map(bad) = max(I(:));
end

detection_map = I > threshold_map;

%% ---- STEP 2: log-domain decision, computed and cross-checked ------------
K = EULER_GAMMA + log(-log(Pfa));     % this is exactly Step 5's K(Pfa)
T_log = c1 + K ./ shape_map;

logdomain = struct();
logdomain.ThresholdMap = T_log;
logdomain.DetectionMap = x > T_log;

mism = logdomain.DetectionMap ~= detection_map;
logdomain.NumMismatch     = sum(mism(:));
logdomain.PercentMismatch = 100 * logdomain.NumMismatch / (h * w);

%% ---- Stats ---------------------------------------------------------------
stats = struct();
stats.NumReferenceCells  = N;
stats.NumDetections      = sum(detection_map(:));
stats.MeanThreshold       = mean(threshold_map(:));
stats.MinShapeParameter  = min(shape_map(:));
stats.MeanShapeParameter = mean(shape_map(:));
stats.MaxShapeParameter  = max(shape_map(:));
stats.FractionAtCMin     = mean(C_raw(:) <= C_MIN + 1e-9);
stats.FractionAtCMax     = mean(C_raw(:) >= C_MAX - 1e-9);
stats.C1Map               = c1;
stats.C2Map               = c2;
stats.ShapeParameterRawMap = C_raw;

if verbose
    fprintf('WeibullCFAR_Floating: %dx%d image, sli=%d, guard=%d, N=%d ref cells, Pfa=%g\n', ...
        h, w, sli, guard, N, Pfa);
    fprintf('  Shape C (clamped) : min=%.3f mean=%.3f max=%.3f  [clamp %.2f, %.2f]\n', ...
        stats.MinShapeParameter, stats.MeanShapeParameter, stats.MaxShapeParameter, C_MIN, C_MAX);
    fprintf('  Clamp saturation  : %.3f%% at CMin, %.3f%% at CMax (of raw C)\n', ...
        100*stats.FractionAtCMin, 100*stats.FractionAtCMax);
    fprintf('  Detections        : %d / %d pixels (%.4f%%)\n', ...
        stats.NumDetections, h*w, 100*stats.NumDetections/(h*w));
    fprintf('  Amplitude vs log-domain mismatch: %d px (%.6f%%) [expect ~0, rounding only]\n', ...
        logdomain.NumMismatch, logdomain.PercentMismatch);
end

end % === end of WeibullCFAR_Floating ======================================


function S = local_box_sum(Xpad, tsli, k)
%LOCAL_BOX_SUM  Sum over a k x k box centred at each pixel of the
%   *unpadded* image, given Xpad which is the original image padded by
%   tsli on every side (tsli must be >= (k-1)/2). Implemented via a
%   summed-area table (integral image) so the whole image is handled with
%   4 index lookups per output pixel, fully vectorised (no explicit
%   nested loop over pixels or window elements).
%
%   Derivation: let P(i,j), i=0..Hp, j=0..Wp, be the 1-indexed prefix sum
%   of Xpad ( P(i,j) = sum(Xpad(1:i,1:j)), P(0,*)=P(*,0)=0 ). The sum over
%   1-indexed inclusive rows [R0,R1], cols [C0,C1] of Xpad is the standard
%   inclusion-exclusion rectangle query:
%       SUM = P(R1,C1) - P(R0-1,C1) - P(R1,C0-1) + P(R0-1,C0-1)
%   II below stores P with an extra leading zero row/col so that
%   II(m,n) = P(m-1,n-1) for all valid m,n (this is exactly the standard
%   "padded integral image" trick, needed because MATLAB has no index 0).
%   Substituting back into the inclusion-exclusion formula gives the four
%   II(...) terms used below.
    [Hp, Wp] = size(Xpad);
    h = Hp - 2*tsli;
    w = Wp - 2*tsli;
    tk = (k - 1) / 2;

    II = cumsum(cumsum(Xpad, 1), 2);
    II = [zeros(1, Wp + 1); zeros(Hp, 1), II];   % II(m,n) = P(m-1,n-1)

    a    = (1:h)';            % 1-indexed output rows, column vector (Hx1)
    b    = (1:w);              % 1-indexed output cols, row vector    (1xW)
    boxR0 = (a + tsli) - tk;   % R0, first row of the box in Xpad (1-indexed)
    boxR1 = (a + tsli) + tk;   % R1, last  row of the box in Xpad (1-indexed)
    boxC0 = (b + tsli) - tk;   % C0
    boxC1 = (b + tsli) + tk;   % C1
    % (named box* to avoid any visual confusion with the outer function's
    % c1 = mean(x) — these are just column/row index bounds, unrelated)

    % Vectorised rectangle query over the whole image at once: MATLAB's
    % A(rowVec, colVec) indexing with two vector subscripts always
    % produces an outer-product-style HxW result, regardless of the
    % vectors' own orientation.
    S = II(boxR1+1, boxC1+1) - II(boxR0, boxC1+1) - II(boxR1+1, boxC0) + II(boxR0, boxC0);
end