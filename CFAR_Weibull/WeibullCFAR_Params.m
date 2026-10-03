function prm = WeibullCFAR_Params(c2, varargin)
%WEIBULLCFAR_PARAMS  Method-of-log-cumulants estimator for Weibull clutter.
%
%   prm = WEIBULLCFAR_PARAMS(c2, ...)
%
%   Weibull clutter model with shape C and scale B:
%       c1 = log(B) - EulerGamma / C
%       c2 = psi(1,1) / C^2                      (psi(1,1) = pi^2/6, exact)
%   so the shape parameter inverts in closed form from c2 alone:
%
%       C = sqrt( psi(1,1) / c2 )
%
%   This matches CFAR_Weibull/legacy/wblmolc.m. B is not formed here: it
%   depends on c1, which the shared additive term carries -- see
%   WeibullCFAR_TLog.
%
%   WHY THIS FILE EXISTS ALONGSIDE WeibullCFAR_Floating.m
%   -----------------------------------------------------
%   WeibullCFAR_Floating.m is the model carried over unmodified from the
%   existing Weibull project -- already verified bit-exact against its RTL
%   and validated on real DE10-Standard hardware. It is kept untouched as
%   the reference. This file re-expresses the same estimator against the
%   shared cfar_front_end so Weibull can be run through the identical
%   pipeline as the other four detectors in the Phase 2 comparison.
%
%   The two must agree, and that agreement is a genuine test of the shared
%   infrastructure rather than a formality: cfar_front_end's c2 is the
%   unbiased k-statistic k2 = (S2 - N*m1^2)/(N-1), which is exactly the
%   sample variance WeibullCFAR_Floating.m computes, and both use symmetric
%   padding and the same guard x guard hole with N = sli^2 - guard^2. Any
%   discrepancy beyond floating-point rounding means the shared front end
%   has drifted from the hardware-validated one. _comparison/verify_models.m
%   checks this on every run.
%
%   NAME-VALUE OPTIONS
%     'CMin' (0.8), 'CMax' (8.0) : clamps on C, matching the defaults of
%       WeibullCFAR_Floating.m and the range the existing c_lut.hex was
%       built over.
%
%   OUTPUT (struct prm)
%     C, C_raw, valid, FractionAtMin, FractionAtMax
%
%   See also: WeibullCFAR_TLog, WeibullCFAR_Floating, cfar_front_end

p = inputParser;
addParameter(p, 'CMin', 0.8, @(v) isnumeric(v) && isscalar(v) && v > 0);
addParameter(p, 'CMax', 8.0, @(v) isnumeric(v) && isscalar(v) && v > 0);
parse(p, varargin{:});
CMin = p.Results.CMin;
CMax = p.Results.CMax;
if CMax <= CMin
    error('WeibullCFAR_Params:BadClamp', 'CMax must exceed CMin.');
end

PSI11 = pi^2 / 6;                 % psi(1,1), exact closed form

C_raw = sqrt(PSI11 ./ c2);
C     = min(max(C_raw, CMin), CMax);

prm = struct();
prm.Name            = 'Weibull';
prm.C               = C;
prm.C_raw           = C_raw;
prm.valid           = isfinite(C) & C > 0;
prm.FractionInvalid = 1 - mean(prm.valid(:));
prm.FractionAtMin   = mean(C_raw(:) <= CMin + 1e-9);
prm.FractionAtMax   = mean(C_raw(:) >= CMax - 1e-9);
prm.CMin = CMin;  prm.CMax = CMax;
end
