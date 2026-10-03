function s = sqrt_fixed(code, fracBits, mantLut)
%SQRT_FIXED  sqrt of a fixed-point value, by leading-zero count + mantissa ROM.
%
%   s = SQRT_FIXED(code, fracBits, mantLut)
%
%   Models the standard hardware construction:
%       normalise code = m * 2^(2j)  with m in [1,4)   (barrel shift by an
%                                                        EVEN amount, so the
%                                                        exponent halves exactly)
%       sqrt(code) = sqrt(m) * 2^j                      (small ROM + shift)
%       sqrt(value) = sqrt(code) * 2^(-fracBits/2)
%
%   ---------------------------------------------------------------------
%   WHY NOT A DIRECT TABLE ON c2
%   ---------------------------------------------------------------------
%   The first version of sqrt_c2_lut addressed c2 uniformly over [0,2] with
%   4096 entries. Measured realised error: 1.5e-2 max -- and it barely improved
%   with table depth (6.3e-2 at 256 entries, 1.1e-2 at 8192), because sqrt is
%   singular at c2 -> 0 and a uniform address cannot resolve it. The worst case
%   sat at c2 = 2.4e-4, where the true sqrt is 0.0155 and the table returned 0.
%
%   That matters because delta is PROPORTIONAL to sqrt(c2) for four of the five
%   detectors (FINDINGS F2/F10), so what the threshold needs is uniform
%   RELATIVE accuracy, not uniform absolute accuracy. The exponent/mantissa
%   split delivers exactly that: the error is bounded by the mantissa step
%   regardless of magnitude, over the whole four-decade range c2 occupies.
%
%   It also costs almost nothing extra, because the leading-zero counter is
%   already in the datapath for log2_fixed (the log-domain shape address).
%
%   INPUTS
%     code     : non-negative integer codes; zero returns zero
%     fracBits : the code's fractional bit count. MUST BE EVEN, so that
%                2^(-fracBits/2) is an exact shift rather than a multiply by
%                sqrt(2) -- checked below rather than assumed.
%     mantLut  : struct loaded from sqrt_mant_lut.mat (m in [1,4) -> sqrt(m))

if mod(fracBits, 2) ~= 0
    error('sqrt_fixed:OddFracBits', ...
        ['fracBits must be even (got %d) so the exponent halves exactly. ' ...
         'An odd width forces a multiply by sqrt(2) in RTL for no benefit.'], fracBits);
end

n = mantLut.entries;

s = zeros(size(code));
pos = code > 0;
if ~any(pos(:)), return; end

c = double(code(pos));

% ---- Normalise to m in [1,4) with an EVEN exponent ----------------------
e = floor(log2(c));
e = e - mod(e, 2);          % force even, so m lands in [1,4)
m = c ./ 2.^e;

% ---- Mantissa ROM, rounded ----------------------------------------------
a = min(max(round((m - 1) / 3 * n), 0), n - 1);

s(pos) = mantLut.requantized(a + 1) .* 2.^(e/2) .* 2^(-fracBits/2);
end
