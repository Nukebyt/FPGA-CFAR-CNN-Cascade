function u = inv_trigamma(y, varargin)
%INV_TRIGAMMA  Inverse of the trigamma function, vectorised.
%
%   u = INV_TRIGAMMA(y)  returns u > 0 such that psi(1,u) == y, elementwise.
%
%   psi(1,.) is strictly DECREASING on (0, inf), mapping (0,inf) -> (0,inf):
%       psi(1,u) -> +inf  as u -> 0+      (psi(1,u) ~ 1/u^2 + 1/u)
%       psi(1,u) -> 0     as u -> +inf    (psi(1,u) ~ 1/u)
%   so the inverse exists and is unique for every y > 0. Elements with
%   y <= 0 (or non-finite) return NaN -- that is a genuine out-of-support
%   condition for the caller to count, not something to silently clamp.
%
%   WHY THIS EXISTS: the G0 detector's method-of-log-cumulants estimator
%   needs exactly this inversion, and the original CFAR_G0/legacy/g0molc.m
%   obtained it with fsolve. The installed MATLAB has no Optimization
%   Toolbox, and in any case a bounded-iteration Newton iteration on a
%   monotone scalar function is what has to go to hardware -- so the
%   floating-point reference and the eventual LUT generator agree on the
%   same root definition rather than on "whatever fsolve converged to".
%
%   _common/invpsi2.m (carried over from the original project) does the same
%   job but is SCALAR-ONLY (it errors on non-scalar input) and uncapped. This
%   function is vectorised over whole HxW parameter maps and has a hard
%   iteration cap, which is what the per-pixel detectors need.
%
%   NAME-VALUE OPTIONS
%     'Tol'     : absolute convergence tolerance on psi(1,u) - y (default 1e-12)
%     'MaxIter' : hard iteration cap (default 60)
%
%   METHOD
%     Initial guess from the asymptotic expansions -- for large y (small u),
%     psi(1,u) ~ 1/u^2 gives u0 = 1/sqrt(y); for small y (large u),
%     psi(1,u) ~ 1/u + 1/(2u^2) gives the exact solution of that quadratic.
%     Then damped Newton using psi(2,.) as the derivative, with each step
%     rejected if it would leave (0, inf).

p = inputParser;
addParameter(p, 'Tol',     1e-12, @(v) isnumeric(v) && isscalar(v) && v > 0);
addParameter(p, 'MaxIter', 60,    @(v) isnumeric(v) && isscalar(v) && v >= 1);
parse(p, varargin{:});
tol     = p.Results.Tol;
maxiter = p.Results.MaxIter;

u = nan(size(y));
ok = isfinite(y) & y > 0;
if ~any(ok(:))
    return;
end

yv = y(ok);

% ---- Initial guess ------------------------------------------------------
% Large y  (u small): psi(1,u) ~ 1/u^2            -> u ~ 1/sqrt(y)
% Small y  (u large): psi(1,u) ~ 1/u + 1/(2u^2)   -> u = (1 + sqrt(1+2y))/(2y)
u0 = zeros(size(yv));
big = yv > 1;
u0(big)  = 1 ./ sqrt(yv(big));
u0(~big) = (1 + sqrt(1 + 2*yv(~big))) ./ (2*yv(~big));

% ---- Damped Newton ------------------------------------------------------
% f(u)  = psi(1,u) - y,   f'(u) = psi(2,u)  (strictly negative, so no
% division-by-zero risk anywhere on the domain).
uv = u0;
active = true(size(uv));
for it = 1:maxiter
    if ~any(active), break; end

    ua = uv(active);
    f  = psi(1, ua) - yv(active);

    done = abs(f) < tol;
    if all(done)
        idx = find(active);
        active(idx(done)) = false;
        continue;
    end

    d    = psi(2, ua);
    step = f ./ d;
    unew = ua - step;

    % Reject any step that leaves the domain; halve it until it is valid.
    % psi(1,.) is convex so plain Newton from either side converges, but a
    % first step from a poor asymptotic guess can still overshoot past 0.
    bad = ~(unew > 0) | ~isfinite(unew);
    for h = 1:40
        if ~any(bad), break; end
        step(bad) = step(bad) / 2;
        unew(bad) = ua(bad) - step(bad);
        bad = ~(unew > 0) | ~isfinite(unew);
    end
    unew(bad) = ua(bad);   % gave up on these; leave them where they were

    uv(active) = unew;
    idx = find(active);
    active(idx(done)) = false;
end

u(ok) = uv;
end
