function z = norminv_local(p)
%NORMINV_LOCAL  Standard-normal quantile, without the Statistics Toolbox.
%
%   z = NORMINV_LOCAL(p) returns z such that Phi(z) == p.
%
%   The installed MATLAB has no Statistics and Machine Learning Toolbox, so
%   norminv() is unavailable; erfcinv() is a core specfun built-in and is
%   used instead.
%
%   Identity:  Phi(z) = erfc(-z/sqrt(2))/2   =>   z = -sqrt(2)*erfcinv(2p)
%
%   The erfcinv form (rather than the algebraically equivalent
%   sqrt(2)*erfinv(2p-1)) is deliberate: the Lognormal detector evaluates
%   this at p = 1-Pfa with Pfa down to 1e-6, where 2p-1 = 1-2e-6 loses
%   precision to cancellation but 2*Pfa does not. Callers wanting the upper
%   tail should pass p = 1-Pfa here only when Pfa is not tiny; for small Pfa
%   use the exact complement directly:
%       z_{1-Pfa} = sqrt(2)*erfcinv(2*Pfa)
%   which is what LognormalCFAR_Params.m does.

z = -sqrt(2) .* erfcinv(2 .* p);
end
