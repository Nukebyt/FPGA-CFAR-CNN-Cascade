function [detection_map, threshold_map, prm, stats, logdomain] = ...
    CACFAR_Floating(I, sli, guard, Pfa, varargin)
%CACFAR_FLOATING  Cell-Averaging CFAR (CA-CFAR), the textbook non-parametric
%   baseline (BC-5) -- the comparator every reviewer expects alongside a
%   parametric-clutter-model comparison.
%
%   [detection_map, threshold_map, prm, stats, logdomain] = ...
%       CACFAR_FLOATING(I, sli, guard, Pfa, ...)
%
%   ---------------------------------------------------------------------
%   WHY THIS IS NOT ANOTHER cfar_compose-BASED DETECTOR
%   ---------------------------------------------------------------------
%   Every one of the six parametric detectors (Weibull ... K) fits a clutter
%   SHAPE from the reference window's log-cumulants, then thresholds via the
%   shared log-domain rule T_log = c1 + delta(shape, Pfa) (finding F1). CA-CFAR
%   fits NOTHING -- it assumes the reference cells are i.i.d. single-look
%   EXPONENTIAL intensity (Rayleigh amplitude) and thresholds directly in the
%   LINEAR domain:
%
%       T_int = alpha * linMean,     alpha = N*(Pfa^(-1/N) - 1)
%       detect  <=>  I_cut > T_int
%
%   alpha is the classical CA-CFAR multiplier (Rohling 1983): under the
%   exponential-clutter assumption, the sum of N i.i.d. reference cells is
%   Gamma(N,.)-distributed, and this alpha is the exact closed-form value
%   that makes P(I_cut > alpha*mean) = Pfa when the assumption holds --
%   nothing to estimate, nothing that can fail to converge. This is exactly
%   why CA-CFAR has no invalid-window failure mode the way the parametric
%   detectors do: it always produces a threshold, at the cost of being wrong
%   whenever the exponential assumption doesn't hold (which, per this
%   project's F6 finding, is often -- CA-CFAR pays for its simplicity with a
%   real accuracy cost, that IS the point of running it as a comparator).
%
%   Reuses cfar_front_end's windowing (identical guard-hole geometry, N
%   reference cells, symmetric padding) via its `linMean` field (added
%   alongside this baseline, see cfar_front_end.m's header) rather than
%   re-deriving the windowing logic -- so this baseline is measured on
%   EXACTLY the same reference cells as the six parametric detectors at a
%   given (sli, guard), not a separately-implemented approximation of them.
%
%   log-domain equivalent (T_log = 0.5*log(T_int+0.5)) is also formed purely
%   for consistency with the other detectors' dual-domain cross-check
%   report -- it is NOT CA-CFAR's canonical domain (linear intensity is),
%   just a diagnostic so `logdomain.PercentMismatch` means the same thing
%   here as it does for the parametric six.
%
%   INPUTS / OUTPUTS: identical contract to WeibullCFAR_Floating.m etc.
%     prm.valid is always true everywhere -- CA-CFAR has no support
%     condition and cannot fail to produce a threshold.
%
%   NAME-VALUE OPTIONS
%     'FrontEnd', 'FrontEndOpts', 'Verbose' -- same meaning as the other
%     detectors' Floating functions.
%
%   See also: cfar_front_end, cfar_compose, CACFAR_Params (registry shim)

p = inputParser;
addParameter(p, 'FrontEnd',     [],  @(v) isempty(v) || isstruct(v));
addParameter(p, 'FrontEndOpts', {},  @iscell);
addParameter(p, 'Verbose',      false, @(v) islogical(v) && isscalar(v));
parse(p, varargin{:});

if ~isscalar(Pfa) || Pfa <= 0 || Pfa >= 1
    error('CACFAR_Floating:BadPfa', 'Pfa must be a scalar in (0,1) -- got %g.', Pfa);
end

if isempty(p.Results.FrontEnd)
    fe = cfar_front_end(I, sli, guard, p.Results.FrontEndOpts{:});
else
    fe = p.Results.FrontEnd;
    if fe.sli ~= sli || fe.guard ~= guard
        error('CACFAR_Floating:FrontEndMismatch', ...
            'Supplied front end was built for sli=%d/guard=%d, not %d/%d.', ...
            fe.sli, fe.guard, sli, guard);
    end
end

N = fe.N;
alpha = N * (Pfa^(-1/N) - 1);

T_int = alpha * fe.linMean;
I = double(I);
detection_map = I > T_int;                 % canonical: linear-domain decision
valid = true(size(I));                     % CA-CFAR never fails to threshold

%% ---- log-domain diagnostic (not the canonical decision) -----------------
T_log = 0.5 * log(max(T_int, 0) + 0.5);
detect_log = (fe.x > T_log) & valid;
mism = detect_log ~= detection_map;

logdomain = struct();
logdomain.ThresholdMap    = T_log;
logdomain.DetectionMap    = detect_log;
logdomain.NumMismatch     = sum(mism(:));
logdomain.PercentMismatch = 100 * logdomain.NumMismatch / numel(mism);

threshold_map = T_int;

prm = struct();
prm.Name  = 'CA-CFAR';
prm.alpha = alpha;
prm.valid = valid;
prm.c1    = fe.c1;
prm.c2    = fe.c2;
prm.c3    = fe.c3;
prm.skew  = fe.skew;
prm.x     = fe.x;

stats = struct();
stats.Detector             = 'CA-CFAR';
stats.NumReferenceCells    = N;
stats.sli                  = sli;
stats.guard                = guard;
stats.Pfa                  = Pfa;
stats.NumDetections        = sum(detection_map(:));
stats.FractionInvalid      = 0;
stats.NumValid             = numel(I);
stats.MeanThresholdLog     = mean(T_log(:), 'omitnan');
stats.LogDomainMismatchPct = logdomain.PercentMismatch;
stats.FractionAtMin        = NaN;
stats.FractionAtMax        = NaN;

if p.Results.Verbose
    fprintf('CACFAR_Floating: %dx%d, sli=%d guard=%d N=%d Pfa=%g alpha=%.4f\n', ...
        size(I,1), size(I,2), sli, guard, N, Pfa, alpha);
    fprintf('  detections : %d (%.4f%%)\n', stats.NumDetections, 100*stats.NumDetections/numel(I));
    fprintf('  log vs linear domain mismatch: %d px (%.6f%%)  [diagnostic only]\n', ...
        logdomain.NumMismatch, logdomain.PercentMismatch);
end
end
