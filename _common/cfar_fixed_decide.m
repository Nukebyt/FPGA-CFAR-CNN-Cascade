function out = cfar_fixed_decide(fe, lut, addrVals, PfaIndex, validMask)
%CFAR_FIXED_DECIDE  The shared back half of every fixed-point detector:
%   address a {Pfa_sel, shape_addr} ROM, add the offset to c1, compare.
%
%   out = CFAR_FIXED_DECIDE(fe, lut, addrVals, PfaIndex, validMask)
%
%   fe        : struct from cfar_front_end_fixed
%   lut       : struct loaded from a <detector>_delta_lut.mat
%   addrVals  : HxW map of the detector's address variable, in its own units
%               (c2 for Weibull/Lognormal, r for Gen. Gamma, s for Burr)
%   PfaIndex  : 1..n_pfa, selects a plane of the ROM
%   validMask : HxW logical, false where the estimator has no solution
%
%   ---------------------------------------------------------------------
%   ADDRESS GENERATION IS A MULTIPLY, NOT A DIVIDE
%   ---------------------------------------------------------------------
%       addr = round( (v - v_min) * (entries-1) / (v_max - v_min) )
%   v_min, v_max and entries are all fixed once the ROM is generated, so
%   (entries-1)/(v_max-v_min) is a single stored constant. In RTL this is
%   subtract, multiply-by-constant, shift, saturate -- no divider.
%
%   Addresses outside the table SATURATE to the end entries. That is correct
%   by construction, not a fallback: every generator sets the address range
%   from the detector's clamp or support boundary, beyond which the exact
%   delta is constant, so the end entry already holds the right value. The
%   saturation counts are returned so a range that is being hit hard shows
%   up rather than hiding.
%
%   ---------------------------------------------------------------------
%   THE DECISION
%   ---------------------------------------------------------------------
%       T_log = c1 + delta      detect = (x > T_log) AND valid
%   Both operands are integers in the same Q-format, so the comparison is a
%   plain signed integer compare -- the whole point of working in the log
%   domain (no exp, no sqrt, no fractional power at decision time).
%
%   OUTPUT (struct out)
%     detection : HxW logical
%     T_log     : HxW dequantised threshold
%     delta     : HxW dequantised offset
%     addr      : HxW integer address actually used
%     SatLow, SatHigh : counts of saturated addresses
%     valid     : the mask applied

n    = lut.entries_per_pfa;
vmin = lut.addr_min;
vmax = lut.addr_max;

if PfaIndex < 1 || PfaIndex > lut.n_pfa
    error('cfar_fixed_decide:BadPfaIndex', ...
        'PfaIndex must be 1..%d (got %d).', lut.n_pfa, PfaIndex);
end

%% ---- Address generation -------------------------------------------------
if isfield(lut,'log_spaced') && lut.log_spaced
    % Log-spaced tables address on log(v). In RTL this is the log-amplitude
    % LUT trick again: a small table maps the exponent+mantissa to an
    % address. Modelled here directly.
    v = max(addrVals, vmin);
    a = round( (log(v) - log(vmin)) / (log(vmax) - log(vmin)) * (n-1) );
else
    a = round( (addrVals - vmin) / (vmax - vmin) * (n-1) );
end

satLow  = a < 0;
satHigh = a > (n-1);
a = min(max(a, 0), n-1);
a(~isfinite(a)) = 0;

%% ---- ROM read -----------------------------------------------------------
flat = (PfaIndex - 1) * n + a + 1;
delta_code = lut.code(flat);
delta = double(delta_code) / lut.format.scale;

%% ---- Threshold and decision --------------------------------------------
T_log = fe.c1 + delta;

if nargin < 5 || isempty(validMask)
    validMask = true(size(T_log));
end
valid = validMask & isfinite(T_log);

detection = (fe.x > T_log) & valid;

out = struct();
out.detection = detection;
out.T_log     = T_log;
out.delta     = delta;
out.addr      = a;
out.SatLow    = sum(satLow(:) & valid(:));
out.SatHigh   = sum(satHigh(:) & valid(:));
out.valid     = valid;
end
