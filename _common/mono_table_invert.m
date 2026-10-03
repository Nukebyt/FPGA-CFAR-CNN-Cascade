function [xi, valid, info] = mono_table_invert(fh, xgrid, y, varargin)
%MONO_TABLE_INVERT  Vectorised inversion of a strictly monotone scalar function.
%
%   [xi, valid] = MONO_TABLE_INVERT(fh, xgrid, y)
%
%   Given a function handle fh that is STRICTLY MONOTONE over xgrid, returns
%   xi (same size as y) with fh(xi) ~= y, elementwise. Elements of y outside
%   the range [min(fh(xgrid)), max(fh(xgrid))] are returned as NaN with the
%   corresponding entry of `valid` false -- those are genuine out-of-support
%   conditions (the estimator has no solution for that sample), and Phase 2
%   counts them rather than clamping them away.
%
%   WHY A TABLE AND NOT A SOLVER
%   ----------------------------
%   Three of the five detectors need a monotone inversion for their shape
%   parameter:
%       Generalized Gamma :  r = 8(k+1)^2/(2k+1)^3   -> k
%       Burr XII          :  s = B(kappa)             -> kappa
%       (G0 uses inv_trigamma, which has a good analytic initial guess and
%        is done with Newton instead.)
%   The originals used fsolve per window. Per-pixel iteration over a
%   500x350 image is both slow and, more importantly, NOT what the hardware
%   will do: in Phase 3 each of these becomes a ROM addressed by a quantised
%   c2/c3 combination. Building a dense monotone table and interpolating is
%   therefore the floating-point model of the actual hardware structure, not
%   a shortcut around it -- the only difference between this and the Phase 3
%   LUT is table density and output word length.
%
%   Interpolation is done in LOG-x when xgrid is strictly positive, because
%   all three of these inverse maps are far closer to linear in log-x than
%   in x, which is also why the Phase 3 tables should be log-addressed.
%
%   NAME-VALUE OPTIONS
%     'Polish'   : true (default) -- run one secant refinement step using fh
%                   itself, so the returned xi is limited by fh's own
%                   accuracy rather than the table's.
%     'LogX'     : 'auto' (default; uses log-x iff all(xgrid > 0)), true, false
%
%   OUTPUT info (diagnostics; cheap, always computed)
%     info.fmin, info.fmax   : the invertible range of fh over xgrid
%     info.Decreasing        : true if fh is decreasing over xgrid
%     info.MaxAbsResidual    : max |fh(xi) - y| over valid entries
%
%   NOTE: monotonicity is CHECKED, not assumed -- a non-monotone fh is an
%   error here, since silently returning one of several roots is exactly the
%   failure mode that makes a per-pixel fsolve untrustworthy.

p = inputParser;
addParameter(p, 'Polish', true,   @(v) islogical(v) && isscalar(v));
addParameter(p, 'LogX',   'auto');
parse(p, varargin{:});
doPolish = p.Results.Polish;
logxOpt  = p.Results.LogX;

xgrid = xgrid(:);
f     = fh(xgrid);

if ~all(isfinite(f))
    error('mono_table_invert:NonFinite', 'fh produced non-finite values on xgrid.');
end
df = diff(f);
if all(df > 0)
    decreasing = false;
elseif all(df < 0)
    decreasing = true;
else
    error('mono_table_invert:NotMonotone', ...
        'fh is not strictly monotone over xgrid (%d sign changes in diff).', ...
        sum(diff(sign(df)) ~= 0));
end

if ischar(logxOpt) || isstring(logxOpt)
    useLogX = all(xgrid > 0);
else
    useLogX = logical(logxOpt);
end
xrep = xgrid;
if useLogX
    xrep = log(xgrid);
end

% interp1 needs ascending sample points
if decreasing
    fq = flipud(f);
    xq = flipud(xrep);
else
    fq = f;
    xq = xrep;
end

fmin = fq(1);
fmax = fq(end);
valid = isfinite(y) & y >= fmin & y <= fmax;

xi = nan(size(y));
if any(valid(:))
    xr = interp1(fq, xq, y(valid), 'linear');
    if useLogX
        xr = exp(xr);
    end

    % ---- One secant polish step against fh itself -----------------------
    % Uses a relative perturbation so it behaves the same across the many
    % decades these parameters span. Any element whose local slope is
    % degenerate is left at its interpolated value.
    if doPolish
        hstep = max(abs(xr) * 1e-6, 1e-12);
        f0 = fh(xr);
        f1 = fh(xr + hstep);
        slope = (f1 - f0) ./ hstep;
        step = (f0 - y(valid)) ./ slope;
        good = isfinite(step) & slope ~= 0;
        xnew = xr;
        xnew(good) = xr(good) - step(good);
        % keep the polish inside the table's own bracket
        lo = min(xgrid); hi = max(xgrid);
        inrange = isfinite(xnew) & xnew >= lo & xnew <= hi;
        xr(inrange) = xnew(inrange);
    end

    xi(valid) = xr;
end

info = struct();
info.fmin = fmin;
info.fmax = fmax;
info.Decreasing = decreasing;
if any(valid(:))
    info.MaxAbsResidual = max(abs(fh(xi(valid)) - y(valid)));
else
    info.MaxAbsResidual = NaN;
end
end
