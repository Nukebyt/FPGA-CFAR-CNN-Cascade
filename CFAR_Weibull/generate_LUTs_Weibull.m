function generate_LUTs_Weibull(varargin)
%GENERATE_LUTS_WEIBULL  Phase 3 LUTs for the Weibull detector.
%
%   GENERATE_LUTS_WEIBULL()
%   GENERATE_LUTS_WEIBULL('Sli',17,'Guard',13,'Pfa',[1e-3 1e-4 1e-5 1e-6])
%
%   Weibull is the REGRESSION ANCHOR for the whole fixed-point phase: its
%   floating-point model is the one carried over from the existing project,
%   verified bit-exact against RTL and validated on real DE10-Standard
%   hardware. If the shared fixed-point front end, the LUT writer and the
%   back-end table are correct, the Weibull fixed-point model must reproduce
%   the Weibull floating model. That is checked by verify_fixedpoint.m.
%
%   ---------------------------------------------------------------------
%   ONE TABLE, NO DIVIDER, NO SQRT
%   ---------------------------------------------------------------------
%       C     = clamp(sqrt(psi1(1)/c2), CMin, CMax)
%       delta = K(Pfa) / C,   K(Pfa) = gamma + log(-log Pfa)
%
%   Both the sqrt and the reciprocal are functions of c2 alone, so the entire
%   back end collapses to a single ROM addressed by {Pfa_sel, c2_addr} whose
%   contents are delta directly. Nothing is computed at run time except the
%   address and one add (T_log = c1 + delta).
%
%   This is deliberately NOT built as "sqrt_c2_lut multiplied by a constant",
%   even though FINDINGS F10 shows delta is exactly proportional to sqrt(c2)
%   in the unclamped region. Folding the clamp into the table is both cheaper
%   (no multiplier) and MORE accurate: the shared sqrt_c2_lut's realised error
%   is worst as c2 -> 0, which is precisely where the C = CMax clamp makes
%   delta constant, so a direct delta table absorbs that error entirely
%   instead of propagating it.
%
%   ---------------------------------------------------------------------
%   ADDRESS RANGE
%   ---------------------------------------------------------------------
%   Chosen from the CLAMP boundaries, not from the empirical c2 spread:
%       C = CMax at c2 = psi1(1)/CMax^2      (below this, delta is constant)
%       C = CMin at c2 = psi1(1)/CMin^2      (above this, delta is constant)
%   Outside that band the table would store the same value repeatedly, so the
%   range is set to it (with a margin) and addresses are saturated to the end
%   entries. This is exact, dataset-independent, and wastes no entries.

p = inputParser;
addParameter(p, 'Sli',     17);
addParameter(p, 'Guard',   13);
addParameter(p, 'Pfa',     [1e-3 1e-4 1e-5 1e-6]);
addParameter(p, 'CMin',    0.8);
addParameter(p, 'CMax',    8.0);
addParameter(p, 'Entries', []);
parse(p, varargin{:});

paths = cfar_setup();
cfg   = fixedpoint_config(p.Results.Sli, p.Results.Guard);
CMin  = p.Results.CMin;  CMax = p.Results.CMax;
PfaL  = p.Results.Pfa;
nEnt  = p.Results.Entries;
if isempty(nEnt), nEnt = cfg.N_SHAPE_ENTRIES; end

lut_dir = fullfile(paths.root, 'lut', 'weibull');
if ~isfolder(lut_dir), mkdir(lut_dir); end

PSI11 = psi(1,1);          % = pi^2/6, exact; psi() avoids shadowing pitfalls
EG    = 0.5772156649015329;

fprintf('=====================================================================\n');
fprintf(' Weibull LUTs  (sli=%d guard=%d N=%d)\n', cfg.sli, cfg.guard, cfg.N);
fprintf('=====================================================================\n');

% ---- Address range from the clamp boundaries ----------------------------
c2_at_CMax = PSI11 / CMax^2;     % below this, C saturates high
c2_at_CMin = PSI11 / CMin^2;     % above this, C saturates low
fprintf('\n  C clamped to [%.2f, %.2f]\n', CMin, CMax);
fprintf('  -> delta varies only for c2 in [%.6f, %.6f]\n', c2_at_CMax, c2_at_CMin);

c2_lo = 0;                        % include 0 so the saturated low end is exact
c2_hi = min(c2_at_CMin * 1.1, cfg.c2.max);
fprintf('  table address range: c2 in [%.6f, %.6f], %d entries\n', c2_lo, c2_hi, nEnt);

K = EG + log(-log(PfaL));
fprintf('\n  K(Pfa) = gamma + log(-log Pfa):\n');
for i = 1:numel(PfaL)
    fprintf('    Pfa=%-8g  K=%+.6f   delta range [%.4f, %.4f]\n', ...
        PfaL(i), K(i), K(i)/CMax, K(i)/CMin);
end

% ---- Build ---------------------------------------------------------------
fprintf('\n--- weibull_delta_lut : {Pfa_sel, c2_addr} -> delta = K/C ---\n');

termFcn = @(c2, pf) ( (EG + log(-log(pf))) ./ ...
                      min(max(sqrt(PSI11 ./ max(c2, realmin)), CMin), CMax) );

% Size the delta format to Weibull's OWN range rather than the shared
% worst case across all five detectors -- see delta_format.m. Weibull's
% delta never exceeds K(min Pfa)/CMin, which is known in closed form.
dmax  = max(K) / CMin;
dfmt  = delta_format(dmax);
fprintf('  delta format: Q%d.%d signed (%d-bit), res %.3e (sized for |delta| <= %.3f)\n', ...
    dfmt.int, dfmt.frac, dfmt.total, dfmt.res, dmax);

addr = struct('var','c2', 'min',c2_lo, 'max',c2_hi, 'entries',nEnt, 'log',false);
info = build_backend_lut(lut_dir, 'weibull_delta_lut', addr, PfaL, termFcn, ...
    dfmt, struct('CMin',CMin, 'CMax',CMax, ...
        'expr','K(Pfa)/clamp(sqrt(psi1(1)/c2),CMin,CMax)', ...
        'sli',cfg.sli, 'guard',cfg.guard));

% ---- Realised lookup error ----------------------------------------------
fprintf('\n  realised lookup error vs exact delta (address rounding included):\n');
c2d = linspace(c2_lo, c2_hi, 200001)';
for ip = 1:numel(PfaL)
    exact = termFcn(c2d, PfaL(ip));
    a = min(max(round((c2d-c2_lo)/(c2_hi-c2_lo)*(nEnt-1)), 0), nEnt-1);
    got = info.requantized((ip-1)*nEnt + a + 1);
    e = got - exact;
    fprintf('    Pfa=%-8g  max|e| = %.3e   RMS = %.3e\n', PfaL(ip), max(abs(e)), sqrt(mean(e.^2)));
end

save(fullfile(lut_dir,'weibull_lut_meta.mat'), 'PfaL','CMin','CMax', ...
     'c2_lo','c2_hi','nEnt','cfg','dfmt');
fprintf('\n=== Weibull LUTs written to %s ===\n', lut_dir);
end
