function [detection_map, threshold_map, prm, stats, logdomain] = ...
    WeibullCFAR_Shared(I, sli, guard, Pfa, varargin)
%WEIBULLCFAR_SHARED  Weibull CFAR detector on the SHARED front end.
%
%   Baseline detector for the Phase 2 comparison. Same algorithm as
%   WeibullCFAR_Floating.m (the hardware-validated model carried over from
%   the existing Weibull project), but built from the shared pieces:
%       cfar_front_end        -- identical to the other four detectors
%       WeibullCFAR_Params    -- C = sqrt(psi(1,1)/c2)
%       WeibullCFAR_TLog      -- delta = K(Pfa)/C
%
%   Running the baseline through the same pipeline as the new detectors is
%   what makes the comparison fair: every detector sees bit-identical
%   c1/c2/c3 from the same padding and the same window geometry, so any
%   difference in Pd or Pfa is attributable to the clutter model and to
%   nothing else.
%
%   WeibullCFAR_Floating.m is left untouched as the reference implementation.
%   _comparison/verify_models.m asserts that this function and that one agree
%   to floating-point rounding, which is a real regression test on the shared
%   front end -- if cfar_front_end ever drifts from the windowing the
%   existing RTL implements, this is where it shows up.
%
%   NAME-VALUE OPTIONS
%     'CMin' (0.8), 'CMax' (8.0)
%     'FrontEnd', 'FrontEndOpts', 'Verbose'
%
%   See also: WeibullCFAR_Floating, WeibullCFAR_Params, WeibullCFAR_TLog

p = inputParser;
addParameter(p, 'CMin',         0.8);
addParameter(p, 'CMax',         8.0);
addParameter(p, 'FrontEnd',     [],  @(v) isempty(v) || isstruct(v));
addParameter(p, 'FrontEndOpts', {},  @iscell);
addParameter(p, 'Verbose',      false, @(v) islogical(v) && isscalar(v));
parse(p, varargin{:});

if isempty(p.Results.FrontEnd)
    feArg = p.Results.FrontEndOpts;
else
    feArg = p.Results.FrontEnd;
end

paramsFcn = @(c2, c3, skew) WeibullCFAR_Params(c2, ...
    'CMin', p.Results.CMin, 'CMax', p.Results.CMax);
tlogFcn = @(prm, pf) WeibullCFAR_TLog(prm, pf);

[detection_map, threshold_map, prm, stats, logdomain] = ...
    cfar_compose(I, sli, guard, Pfa, paramsFcn, tlogFcn, 'Weibull', feArg);

if p.Results.Verbose
    fprintf('WeibullCFAR_Shared: %dx%d, sli=%d guard=%d N=%d Pfa=%g\n', ...
        size(I,1), size(I,2), sli, guard, stats.NumReferenceCells, Pfa);
    fprintf('  C (clamped) : min=%.3f mean=%.3f max=%.3f  [%.2f, %.2f]\n', ...
        min(prm.C(:)), mean(prm.C(:)), max(prm.C(:)), prm.CMin, prm.CMax);
    fprintf('  clamp saturation: %.3f%% low, %.3f%% high\n', ...
        100*prm.FractionAtMin, 100*prm.FractionAtMax);
    fprintf('  detections : %d (%.4f%%)\n', ...
        stats.NumDetections, 100*stats.NumDetections/numel(I));
    fprintf('  log vs amplitude domain mismatch: %d px (%.6f%%)\n', ...
        logdomain.NumMismatch, logdomain.PercentMismatch);
end
end
