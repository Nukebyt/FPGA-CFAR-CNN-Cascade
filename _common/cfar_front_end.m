function fe = cfar_front_end(I, sli, guard, varargin)
%CFAR_FRONT_END  Shared windowed log-cumulant front end for ALL five detectors.
%
%   fe = CFAR_FRONT_END(I, sli, guard, ...)
%
%   This is the block that is IDENTICAL across Weibull / Lognormal /
%   Generalized-Gamma / G0 / Burr-XII. In the streaming FPGA architecture it
%   is the line buffer + window-sum + moment accumulators; everything after
%   it is a small per-distribution ROM. Isolating it here means:
%
%     * the cross-detector comparison (Phase 2) compares ONLY the back ends,
%       on bit-identical c1/c2/c3 inputs -- no detector gets an advantage
%       from a different windowing or padding convention;
%     * the front end is written once, verified once, and synthesised once.
%
%   ---------------------------------------------------------------------
%   WHAT IT COMPUTES
%   ---------------------------------------------------------------------
%       I_amp = sqrt(I + 0.5)              0.5-LSB dequantization offset
%       x     = log(I_amp)                 log-amplitude
%
%   Over the reference cells -- the sli x sli window around each CUT minus a
%   concentric guard x guard hole, N = sli^2 - guard^2 cells -- it returns the
%   first three sample log-cumulants c1, c2, c3.
%
%   ---------------------------------------------------------------------
%   THE CENTRING CONSTANT (X0) -- why it exists
%   ---------------------------------------------------------------------
%   c2 and c3 are computed from raw power sums (S1, S2, S3). For 8-bit SAR
%   imagery x lies in [-0.347, 2.772] with a typical mean near 2, while c3 is
%   often ~1e-2 or smaller. Forming c3 as S3/N - 3*m1*S2/N + 2*m1^3 then
%   subtracts quantities of order 8 to leave a result of order 0.01 --
%   roughly three decimal digits of cancellation, tolerable in double
%   precision but FATAL in the Q-format fixed-point model of Phase 3.
%
%   Central moments are shift-invariant, so all sums are taken on (x - X0)
%   instead. X0 defaults to a FIXED CONSTANT (the midpoint of the 8-bit x
%   range), not a per-image mean, precisely so the hardware can reproduce it
%   exactly: subtracting a constant from x is free in hardware because it is
%   simply baked into the 256-entry log-amplitude LUT at generation time.
%   c1 has X0 added back before it is returned, so callers never see it.
%
%   ---------------------------------------------------------------------
%   INPUTS
%     I     : HxW SAR image, 8-bit-valued intensity (double or uint8)
%     sli   : sliding window size, odd, >= 3
%     guard : guard region size,  odd, >= 1, < sli
%
%   NAME-VALUE OPTIONS
%     'X0'       : centring constant (default 1.2125, the 8-bit x midpoint)
%     'Unbiased' : true (default) -> c2, c3 are the unbiased k-statistics
%                   k2, k3; false -> the raw central moments m2, m3 used by
%                   the original legacy/ scripts.
%                   NOTE k2 == the c2 of WeibullCFAR_Floating.m exactly
%                   (both are (S2 - N*m1^2)/(N-1)), so the Weibull baseline
%                   stays numerically consistent with the existing project.
%     'C2Floor'  : floor applied to c2 (default 1e-9), guards against a
%                   negative variance from floating-point cancellation on a
%                   perfectly flat window.
%     'IFloor'   : lower clamp applied to I before the log (default 0, i.e.
%                   off). See "THE DARK-PIXEL PROBLEM" below -- this is a
%                   one-entry change to the hardware log LUT and costs
%                   nothing at run time, but it materially changes c3.
%
%   ---------------------------------------------------------------------
%   THE DARK-PIXEL PROBLEM (why 'IFloor' exists)
%   ---------------------------------------------------------------------
%   x = log(sqrt(I+0.5)) is extremely steep at the bottom of the 8-bit range:
%       I = 0 -> x = -0.347      I = 1 -> x = +0.203      I = 2 -> x = +0.458
%   while typical SSDD clutter sits near x ~ 2. A SINGLE I=0 pixel inside a
%   216-cell reference window is therefore a ~2.3-unit outlier, and the third
%   central moment weights it by the CUBE. c3 is consequently far more
%   sensitive to a handful of near-black pixels than c2 is.
%
%   This is not hypothetical. Measured across datasets at sli=21/guard=15
%   (_comparison/compare_datasets.m, Results/dataset_comparison.csv):
%
%       dataset                       median log-skew   % outside Burr support
%       MSTAR, native complex SAR          -0.413                0.4%
%       MSTAR, rounded to 8 bits           -0.725                4.5%
%       SSDD, 8-bit JPEG                   -1.420               56.1%
%
%   The physics is the same X-band SAR amplitude in all three rows; only the
%   ENCODING differs. Rounding to 8 bits alone moves the skewness measurably
%   but modestly; the much larger remaining gap belongs to SSDD's lossy JPEG
%   compression, whose ringing scatters isolated near-black pixels through
%   otherwise uniform clutter. (Note MSTAR actually contains MORE dark pixels
%   overall -- large genuinely-dark background regions -- and is barely
%   affected: a uniform dark area moves the window mean, not its skewness.
%   It is ISOLATED dark outliers that wreck c3.)
%
%   Setting 'IFloor' to 1 or 2 clamps I up before the log and collapses the
%   worst of that cliff. It is left OFF by default so the headline comparison
%   reflects the data exactly as supplied; its measured effect is in
%   _comparison/run_darkpixel_study.m. In hardware it is free -- it only
%   rewrites the bottom few entries of the 256-entry log-amplitude ROM and
%   changes no logic at all.
%
%   OUTPUT (struct fe)
%     x      : HxW log-amplitude image (the CUT values the decision uses)
%     c1     : HxW mean of x over reference cells
%     c2     : HxW variance of x over reference cells
%     c3     : HxW third central moment / third cumulant of x
%     skew   : HxW c3 ./ c2.^1.5  -- the DIMENSIONLESS log-skewness.
%              Both 3-parameter detectors (Generalized Gamma via r = skew^2,
%              Burr XII via s = skew) depend on the shape parameter through
%              this single scalar and nothing else, which is what collapses
%              their otherwise-2-D (c2,c3) shape tables to 1-D. Returned
%              here so Phase 3's LUT range analysis can use it directly.
%     linMean: HxW LINEAR-domain (intensity) mean over reference cells --
%              NOT log-amplitude, and NOT the same quantity as exp(c1) in
%              general (E[log X] != log E[X]). Added for CA-CFAR (BC-5):
%              the textbook cell-averaging baseline compares the raw
%              intensity of the cell under test against a linear reference-
%              cell average times a Pfa-derived multiplier -- it does not
%              fit any clutter shape at all, so it needs the plain windowed
%              mean this field provides, not the log-cumulants above. Every
%              one of the six parametric detectors ignores this field;
%              added so CA-CFAR can reuse this front end's verified
%              windowing/guard-hole geometry instead of re-deriving it.
%     N      : number of reference cells, sli^2 - guard^2
%     sli, guard, X0, Unbiased : the settings used
%
%   See also: WeibullCFAR_Floating, LognormalCFAR_Floating, GenGammaCFAR_Floating,
%             G0CFAR_Floating, BurrCFAR_Floating

%% ---- Options -----------------------------------------------------------
p = inputParser;
addParameter(p, 'X0',       1.2125, @(v) isnumeric(v) && isscalar(v));
addParameter(p, 'Unbiased', true,   @(v) islogical(v) && isscalar(v));
addParameter(p, 'C2Floor',  1e-9,   @(v) isnumeric(v) && isscalar(v) && v >= 0);
addParameter(p, 'IFloor',   0,      @(v) isnumeric(v) && isscalar(v) && v >= 0);
parse(p, varargin{:});
X0       = p.Results.X0;
unbiased = p.Results.Unbiased;
c2floor  = p.Results.C2Floor;
ifloor   = p.Results.IFloor;

%% ---- Validate ----------------------------------------------------------
if mod(sli,2) == 0 || sli < 3
    error('cfar_front_end:BadWindow', 'sli must be odd and >= 3 (got %g).', sli);
end
if mod(guard,2) == 0 || guard < 1 || guard >= sli
    error('cfar_front_end:BadGuard', 'guard must be odd, >= 1 and < sli (got %g).', guard);
end

N = sli^2 - guard^2;
if unbiased && N < 3
    error('cfar_front_end:TooFewRefCells', ...
        'Unbiased c3 (k3) needs N >= 3 reference cells; sli=%d/guard=%d gives N=%d.', ...
        sli, guard, N);
end

I = double(I);
tsli = (sli - 1) / 2;

%% ---- Log-amplitude, centred --------------------------------------------
if ifloor > 0
    I = max(I, ifloor);      % free in hardware: it only rewrites LUT entries
end                           % 0..ifloor-1 to all hold LUT(ifloor)
x = log(sqrt(I + 0.5));
xc = x - X0;                      % centred; central moments are unaffected

%% ---- Reference-cell power sums via integral images ---------------------
% Sums over "sli x sli box minus concentric guard x guard box" are formed as
% (sli-box sum) - (guard-box sum). A summed-area table gives each box sum in
% O(1) per pixel after one O(h*w) precompute -- the vectorised analogue of
% the incremental add-new-edge/drop-old-edge accumulator the RTL uses.
xp  = padarray(xc,    [tsli, tsli], 'symmetric', 'both');
xp2 = xp .^ 2;
xp3 = xp .^ 3;

S1 = box_sum(xp,  tsli, sli) - box_sum(xp,  tsli, guard);
S2 = box_sum(xp2, tsli, sli) - box_sum(xp2, tsli, guard);
S3 = box_sum(xp3, tsli, sli) - box_sum(xp3, tsli, guard);

%% ---- Linear-domain (intensity) reference-cell mean, for CA-CFAR --------
Ip = padarray(I, [tsli, tsli], 'symmetric', 'both');
Slin = box_sum(Ip, tsli, sli) - box_sum(Ip, tsli, guard);
linMean = Slin / N;

%% ---- Central moments ---------------------------------------------------
m1 = S1 / N;
m2 = S2 / N - m1.^2;
m3 = S3 / N - 3 * m1 .* (S2 / N) + 2 * m1.^3;

if unbiased
    % k-statistics: unbiased estimators of the population cumulants.
    c2 = (N / (N - 1))                 * m2;
    c3 = (N^2 / ((N - 1) * (N - 2)))   * m3;
else
    c2 = m2;
    c3 = m3;
end

c2 = max(c2, c2floor);

%% ---- Outputs -----------------------------------------------------------
fe = struct();
fe.x        = x;
fe.c1       = m1 + X0;            % undo the centring shift
fe.c2       = c2;
fe.c3       = c3;
fe.skew     = c3 ./ c2.^1.5;
fe.linMean  = linMean;
fe.N        = N;
fe.sli      = sli;
fe.guard    = guard;
fe.X0       = X0;
fe.IFloor   = ifloor;
fe.Unbiased = unbiased;

end % === cfar_front_end ===================================================


function S = box_sum(Xpad, tsli, k)
%BOX_SUM  Sum over a k x k box centred at every pixel of the unpadded image,
%   given Xpad = the image padded by tsli on all sides (tsli >= (k-1)/2).
%   Standard padded summed-area table: II(m,n) = sum(Xpad(1:m-1, 1:n-1)), so
%   an inclusive rectangle [R0..R1] x [C0..C1] of Xpad is
%       II(R1+1,C1+1) - II(R0,C1+1) - II(R1+1,C0) + II(R0,C0).
    [Hp, Wp] = size(Xpad);
    h = Hp - 2*tsli;
    w = Wp - 2*tsli;
    tk = (k - 1) / 2;

    II = cumsum(cumsum(Xpad, 1), 2);
    II = [zeros(1, Wp + 1); zeros(Hp, 1), II];

    r = (1:h)' + tsli;
    c = (1:w)  + tsli;
    R0 = r - tk;  R1 = r + tk;
    C0 = c - tk;  C1 = c + tk;

    S = II(R1+1, C1+1) - II(R0, C1+1) - II(R1+1, C0) + II(R0, C0);
end
