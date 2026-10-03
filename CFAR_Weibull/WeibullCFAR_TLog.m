function [delta, info] = WeibullCFAR_TLog(prm, Pfa)
%WEIBULLCFAR_TLOG  Log-domain threshold offset for Weibull clutter.
%
%   [delta, info] = WEIBULLCFAR_TLOG(prm, Pfa)
%
%   Returns delta = T_log - c1 for the decision  x_cut > c1 + delta.
%
%   DERIVATION (identical to the one inside WeibullCFAR_Floating.m Step 2)
%       B     = exp(c1 + EulerGamma/C)
%       T_amp = B * (-log(Pfa))^(1/C)
%   =>  log(T_amp) = c1 + EulerGamma/C + log(-log Pfa)/C
%                  = c1 + [EulerGamma + log(-log Pfa)] / C
%                  = c1 + K(Pfa)/C
%
%       delta = K(Pfa) / C,     K(Pfa) = EulerGamma + log(-log(Pfa))
%
%   K is a single constant per Pfa -- it is what the existing project's
%   pfa_constant_lut.hex stores (4 entries for Pfa in {1e-3,1e-4,1e-5,1e-6}),
%   and K/C precomputed over the quantised C grid is what kc_lut.hex stores,
%   which is how the runtime divider was removed from the Weibull datapath.
%
%   Note B is formed from the CLAMPED C, not the raw one -- see the
%   "IMPORTANT DEVIATION" note in WeibullCFAR_Floating.m's header. Since C
%   only ever enters here through the already-clamped prm.C, that behaviour
%   is inherited automatically.
%
%   Pfa must be below 1/e for K to be defined with a real logarithm:
%   -log(Pfa) > 1 requires Pfa < exp(-1) = 0.3679. Every Pfa this project
%   uses is orders of magnitude below that.
%
%   INPUTS
%     prm : struct from WeibullCFAR_Params (field C)
%     Pfa : scalar, 0 < Pfa < 1
%
%   OUTPUTS
%     delta : HxW map of T_log - c1
%     info  : struct with the scalar K used (the Phase 3 LUT entry)

if ~isscalar(Pfa) || Pfa <= 0 || Pfa >= 1
    error('WeibullCFAR_TLog:BadPfa', 'Pfa must be a scalar in (0,1) -- got %g.', Pfa);
end

EULER_GAMMA = 0.5772156649015329;
K = EULER_GAMMA + log(-log(Pfa));

delta = K ./ prm.C;

info = struct();
info.K   = K;
info.Pfa = Pfa;
end
