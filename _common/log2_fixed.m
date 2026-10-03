function lg = log2_fixed(code, fracBits, mantLut)
%LOG2_FIXED  log2 of a fixed-point value, by leading-zero count + mantissa ROM.
%
%   lg = LOG2_FIXED(code, fracBits, mantLut)
%
%   Models the standard hardware construction, exactly:
%       e = position of the most significant set bit   (priority encoder)
%       m = code / 2^e   in [1, 2)                     (barrel shift)
%       log2(code) = e + mantLut[m]                    (small ROM)
%       log2(value) = log2(code) - fracBits            (constant subtract)
%
%   No divider, no iteration, no floating point. The error is bounded by the
%   mantissa table's address step REGARDLESS OF MAGNITUDE -- which is the
%   whole reason this construction is used instead of a direct table on the
%   value. A uniformly-addressed log table over a quantity spanning several
%   decades (c2 here spans about four) collapses in accuracy at the low end;
%   this does not.
%
%   INPUTS
%     code     : integer codes (non-negative). Zero and negative map to -Inf,
%                which callers must treat as an invalid window rather than
%                propagate -- log2(0) has no fixed-point representation and
%                silently substituting a finite value here would put a
%                garbage address on the shape ROM.
%     fracBits : the code's fractional bit count, so value = code/2^fracBits
%     mantLut  : struct loaded from log2_mant_lut.mat
%
%   OUTPUT
%     lg : log2 of the VALUE (not the code), same size as code, -Inf where
%          code <= 0.

n = mantLut.entries;

lg = -inf(size(code));
pos = code > 0;
if ~any(pos(:)), return; end

c = double(code(pos));

% ---- Exponent: index of the MSB (a priority encoder in RTL) -------------
e = floor(log2(c));

% ---- Mantissa: normalise into [1,2) (a barrel shift in RTL) -------------
m = c ./ 2.^e;

% ---- Mantissa ROM lookup, with rounding ---------------------------------
% Rounding rather than truncating the mantissa address halves the worst-case
% error and costs a single +0.5-LSB add before the truncation in RTL.
a = min(max(round((m - 1) * n), 0), n - 1);

lg(pos) = e + mantLut.requantized(a + 1) - fracBits;
end
