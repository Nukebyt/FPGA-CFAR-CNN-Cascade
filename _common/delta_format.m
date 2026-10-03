function f = delta_format(maxAbsDelta, totalBits)
%DELTA_FORMAT  Size the back-end offset format to a detector's actual range.
%
%   f = DELTA_FORMAT(maxAbsDelta)            % 16-bit default
%   f = DELTA_FORMAT(maxAbsDelta, totalBits)
%
%   delta = T_log - c1 spans wildly different ranges across the five
%   detectors (measured over SSDD at sli=17/guard=13, Pfa down to 1e-6):
%
%       Weibull       max  4.0     -> 3 integer bits
%       Lognormal     max  6.6     -> 3
%       Gen. Gamma    max 12.8     -> 4
%       G0            max 12.8     -> 4
%       Burr XII      max 30.5     -> 5
%
%   A single shared format has to accommodate Burr's 5 integer bits, which
%   costs every other detector two bits of fractional precision for range it
%   never uses. Since the realised LUT error is dominated by the address step
%   rather than the stored value's quantisation, those two bits are not free:
%   at 16 bits total, Q3.12 resolves 2.44e-4 where Q5.10 resolves 9.77e-4.
%
%   The comparator downstream still works on the SHARED T_log format, so
%   giving each detector its own delta format costs nothing in the datapath --
%   only the ROM contents and one re-scale differ.
%
%   A 25% margin is applied above the measured maximum, so a window slightly
%   more extreme than anything in the sample does not overflow the ROM. That
%   margin is deliberately modest: lut_write treats overflow as an error
%   rather than saturating, so an under-sized format fails loudly at
%   generation time, not silently on hardware.

if nargin < 2, totalBits = 16; end

MARGIN = 1.25;
need   = maxAbsDelta * MARGIN;

intBits = max(0, ceil(log2(max(need, eps))));
fracBits = totalBits - intBits - 1;          % signed

if fracBits < 4
    error('delta_format:TooWide', ...
        ['delta range +-%.3g needs %d integer bits, leaving only %d ' ...
         'fractional bits at %d total. Widen totalBits.'], ...
        maxAbsDelta, intBits, fracBits, totalBits);
end

f = struct();
f.int    = intBits;
f.frac   = fracBits;
f.signed = true;
f.total  = totalBits;
f.scale  = 2^fracBits;
f.res    = 2^-fracBits;
f.min    = -2^intBits;
f.max    =  2^intBits - f.res;
f.sizedFor = maxAbsDelta;
end
