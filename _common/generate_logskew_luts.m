function generate_logskew_luts(varargin)
%GENERATE_LOGSKEW_LUTS  The shared address generator for BOTH 3-parameter
%   detectors, built without a divider.
%
%   GENERATE_LOGSKEW_LUTS()
%
%   ---------------------------------------------------------------------
%   THE PROBLEM
%   ---------------------------------------------------------------------
%   Generalized Gamma addresses its shape table by  r = c3^2 / c2^3, and
%   Burr XII by  s = c3 / c2^1.5. Both are RATIOS of cumulants, and computing
%   either directly needs a division and a fractional power -- exactly the
%   operations the whole LUT architecture exists to remove.
%
%   ---------------------------------------------------------------------
%   THE SOLUTION, AND A UNIFICATION THAT FALLS OUT OF IT
%   ---------------------------------------------------------------------
%   In the log domain both ratios become linear:
%
%       log2|s| = log2|c3| - 1.5 * log2(c2)
%       log2(r) = 2 * log2|s|                       since r = s^2
%
%   So a single quantity -- log2|s|, plus the sign bit of c3 -- addresses
%   BOTH detectors. Generalized Gamma's address is just Burr's scaled by two,
%   which is a shift, not a computation. **One address generator, two ROMs.**
%
%   That leaves only log2 of a fixed-point number to implement, which is the
%   standard leading-zero-count construction and needs no divider either:
%
%       for an integer code v > 0 :  e = floor(log2 v)          (a priority
%                                    m = v / 2^e  in [1,2)       encoder)
%       log2(v) = e + log2(m)                                    (small ROM)
%
%   A 256-entry table over the mantissa gives log2(m) to ~1e-5. The exponent
%   comes from a leading-zero counter, which is combinational logic, not
%   arithmetic. This file builds that mantissa table.
%
%   ---------------------------------------------------------------------
%   WHY LOG-ADDRESSING IS ALSO THE RIGHT *STATISTICAL* CHOICE
%   ---------------------------------------------------------------------
%   Independently of the divider argument: r spans roughly [0.02, 6.6] over
%   the useful k range, a ~300x span, and the map r -> k is violently
%   non-linear at both ends. A uniformly-spaced r address would spend most of
%   its entries where k barely moves and starve the region where it moves
%   fastest. Log spacing matches the curvature, so the same table depth buys
%   substantially lower worst-case error.
%
%   OUTPUT: lut/shared/log2_mant_lut.{mat,txt,hex}

p = inputParser;
addParameter(p, 'Sli',     17);
addParameter(p, 'Guard',   13);
addParameter(p, 'Entries', 1024);
parse(p, varargin{:});

paths = cfar_setup();
cfg   = fixedpoint_config(p.Results.Sli, p.Results.Guard);
lut_dir = fullfile(paths.root, 'lut', 'shared');
if ~isfolder(lut_dir), mkdir(lut_dir); end

n = p.Results.Entries;

fprintf('=====================================================================\n');
fprintf(' Shared log-skewness address LUT\n');
fprintf('=====================================================================\n');
fprintf('\n--- log2_mant_lut : m in [1,2) -> log2(m) ---\n');

% Mantissa grid. Address = the fractional bits of the normalised value, so
% the grid is uniform in m over [1, 2) -- which is what a barrel shifter
% produces directly, no scaling needed.
m = (1 + (0:n-1)'/n);
lg = log2(m);

% log2(m) in [0,1) -> unsigned, 0 integer bits. 16 bits all fractional gives
% ~1.5e-5, comfortably below the shape table's own address step.
fmt = struct('int',0, 'frac',16, 'signed',false, 'total',16, ...
             'scale',2^16, 'res',2^-16, 'min',0, 'max',1-2^-16);

info = lut_write(lut_dir, 'log2_mant_lut', lg, fmt, struct( ...
    'entries', n, 'expr', 'log2(m), m in [1,2) uniform', ...
    'usage', 'log2(v) = exponent_from_LZC + this_table[mantissa]'));

% ---- Realised error over the full fixed-point range ---------------------
% Sweep actual c2 codes and check the reconstructed log2.
fprintf('\n  realised log2 error over the c2 code range:\n');
codes = unique(round(logspace(0, log10(2^15-1), 20000)))';
e = floor(log2(codes));
mant = codes ./ 2.^e;
% ROUND, not floor: rounding the mantissa costs a +0.5-LSB add before
% truncation in RTL and halves the worst-case error for free.
addr = min(max(round((mant - 1) * n), 0), n-1);
lg_hat = e + info.requantized(addr+1);
err = lg_hat - log2(codes);
fprintf('    max |e| = %.3e   RMS = %.3e   (in log2 units)\n', ...
    max(abs(err)), sqrt(mean(err.^2)));
fprintf('    -> worst relative error in the reconstructed value: %.3e\n', ...
    max(abs(2.^err - 1)));

fprintf('\n  NOTE the error is bounded by the mantissa step (1/%d) regardless of\n', n);
fprintf('  magnitude -- that is the point of the exponent/mantissa split, and it\n');
fprintf('  is why this handles c2 spanning several decades without the accuracy\n');
fprintf('  collapse a uniformly-addressed table shows near zero.\n');

%% =====================================================================
%% sqrt mantissa table -- same construction, same leading-zero counter
%% =====================================================================
fprintf('\n--- sqrt_mant_lut : m in [1,4) -> sqrt(m) ---\n');

% m spans [1,4) rather than [1,2) because the exponent is forced EVEN so it
% can be halved exactly (see sqrt_fixed.m). sqrt(m) then lands in [1,2).
% Half-open [1,4): the last entry is just under 4, never 4 itself. That is
% what a barrel shifter actually produces (the mantissa's fractional bits
% cannot reach the next power of two), and it keeps sqrt(m) < 2 so the Q1.14
% output format holds. lut_write's overflow guard caught the closed-interval
% version, which put sqrt(4)=2.0 in the last entry.
ms  = 1 + 3*(0:n-1)'/n;
sq  = sqrt(ms);

sfmt = struct('int',1, 'frac',14, 'signed',false, 'total',15, ...
              'scale',2^14, 'res',2^-14, 'min',0, 'max',2-2^-14);

sinfo = lut_write(lut_dir, 'sqrt_mant_lut', sq, sfmt, struct( ...
    'entries', n, 'expr', 'sqrt(m), m in [1,4) uniform', ...
    'usage', 'sqrt(v) = sqrt_mant[mantissa] * 2^(exp/2), exponent forced even'));

% ---- Realised error, compared against the old uniform-address table -----
fprintf('\n  realised sqrt error over the c2 code range (exponent/mantissa):\n');
codes = unique(round(logspace(0, log10(2^25-1), 40000)))';
FR = 24;                                  % c2 fractional bits in the config
got = sqrt_fixed(codes, FR, sinfo_as_lut(sinfo, n));
exact = sqrt(codes / 2^FR);
rel = abs(got - exact) ./ exact;
fprintf('    max |rel err| = %.3e   RMS rel = %.3e\n', max(rel), sqrt(mean(rel.^2)));
fprintf('    (a uniformly-addressed 4096-entry table on c2 in [0,2] measured\n');
fprintf('     1.5e-2 ABSOLUTE worst case, unbounded in relative terms near 0)\n');

fprintf('\n=== log-skew and sqrt address LUTs written to %s ===\n', lut_dir);
end

function L = sinfo_as_lut(info, n)
%SINFO_AS_LUT  Wrap a fresh lut_write result in the shape sqrt_fixed expects,
%   so the error check exercises the exact same code path the detector will.
    L = struct('entries', n, 'requantized', info.requantized);
end
