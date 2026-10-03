function dets = detector_registry(varargin)
%DETECTOR_REGISTRY  The set of detectors the Phase 2 comparison sweeps.
%
%   dets = DETECTOR_REGISTRY()                all detectors
%   dets = DETECTOR_REGISTRY('Set','core')    the five headline models
%
%   Returns a struct array with fields
%     .name    short label used in results tables and figure legends
%     .fcn     @(I, sli, guard, Pfa, varargin) -> the standard 5-output
%              detector contract. Convenient for one-off calls and used by
%              verify_models.m.
%     .params  @(c2, c3, skew) -> parameter struct, or [] (see CA-CFAR below)
%     .tlog    @(prm, Pfa)     -> delta = T_log - c1, or [] (see CA-CFAR below)
%     .nmom    2 or 3 -- how many log-cumulants the estimator consumes (0 for
%              CA-CFAR, which fits no clutter shape at all)
%     .group   'core', 'variant', or 'baseline'
%     .note    one line on what the entry is for
%
%   CA-CFAR HAS params=[] AND tlog=[]
%   ----------------------------------
%   Unlike the six parametric detectors, CA-CFAR (BC-5, the textbook
%   non-parametric comparator) doesn't factor into a Pfa-independent
%   estimator step at all -- see CACFAR_Floating.m's header for why its
%   decision rule doesn't fit the shared c1+delta form. Sweep scripts
%   (run_comparison.m/run_comparison_hrsid.m) check for params==[] and fall
%   back to calling .fcn directly per (image, sli, guard, Pfa) for such
%   entries, instead of the factored front-end/estimator/threshold path.
%   This costs nothing here since CA-CFAR's ".estimator" (a windowed mean,
%   already computed by the shared front end) is free -- the factoring
%   exists to avoid re-running expensive iterative solves (G0's bisection
%   above all), which CA-CFAR doesn't have.
%
%   WHY params AND tlog ARE EXPOSED SEPARATELY
%   ------------------------------------------
%   The Phase 2 sweep is a 4-way grid over (image, window size, detector,
%   Pfa), and the work does NOT factor evenly across it:
%     * cfar_front_end depends on (image, sli, guard) only -- it is shared by
%       every detector and every Pfa;
%     * the ESTIMATOR depends on (image, sli, guard, detector) -- shared
%       across all four Pfa values;
%     * only the THRESHOLD depends on Pfa.
%   Calling the monolithic .fcn inside the Pfa loop would recompute the front
%   end 28 times and each estimator 4 times per image per window size. Since
%   the estimators are the expensive part (G0's bisection above all),
%   splitting the registry this way is what makes the full grid tractable --
%   and it is also the split the hardware has, so the timing breakdown the
%   sweep reports maps directly onto the FPGA block diagram.
%
%   .nmom is the number that decides FPGA cost: a 3-moment detector needs a
%   third accumulator chain and more DSP in window_sum, which is the
%   identified critical-path risk for Phase 4.
%
%   Variants are included because they change the hardware cost by more than
%   the model choice does:
%     G0-L1          : 1-D shape ROM instead of a 2-D one, but with a support
%                      condition that fails on homogeneous clutter
%     GenGamma-exact : inverts the true MoLC relation rather than the cubic
%                      asymptotic approximation the supplied formula uses

p = inputParser;
addParameter(p, 'Set', 'all', @(v) any(strcmpi(v, {'all','core','variant','baseline'})));
parse(p, varargin{:});
want = lower(p.Results.Set);

d = struct('name', {}, 'fcn', {}, 'params', {}, 'tlog', {}, ...
           'nmom', {}, 'group', {}, 'note', {});

d(end+1) = entry('Weibull', ...
    @(I,s,g,pf,varargin) WeibullCFAR_Shared(I,s,g,pf,varargin{:}), ...
    @(c2,c3,sk) WeibullCFAR_Params(c2), ...
    @(prm,pf)   WeibullCFAR_TLog(prm,pf), ...
    2, 'core', 'Baseline: the model already built and validated on DE10-Standard hardware');

d(end+1) = entry('Lognormal', ...
    @(I,s,g,pf,varargin) LognormalCFAR_Floating(I,s,g,pf,varargin{:}), ...
    @(c2,c3,sk) LognormalCFAR_Params(c2), ...
    @(prm,pf)   LognormalCFAR_TLog(prm,pf), ...
    2, 'core', 'Cheapest possible back end: one sqrt and one constant multiply, no divider');

d(end+1) = entry('GenGamma', ...
    @(I,s,g,pf,varargin) GenGammaCFAR_Floating(I,s,g,pf,'Solver','cubic',varargin{:}), ...
    @(c2,c3,sk) GenGammaCFAR_Params(c2,c3,'Solver','cubic'), ...
    @(prm,pf)   GenGammaCFAR_TLog(prm,pf), ...
    3, 'core', 'Supplied formula: cubic (asymptotic) solve for k');

d(end+1) = entry('G0', ...
    @(I,s,g,pf,varargin) G0CFAR_Floating(I,s,g,pf,'Mode','LA',varargin{:}), ...
    @(c2,c3,sk) G0CFAR_Params(c2,c3,'Mode','LA'), ...
    @(prm,pf)   G0CFAR_TLog(prm,pf), ...
    3, 'core', 'Both L and alpha estimated, matching legacy/g0molc.m');

d(end+1) = entry('BurrXII', ...
    @(I,s,g,pf,varargin) BurrCFAR_Floating(I,s,g,pf,varargin{:}), ...
    @(c2,c3,sk) BurrCFAR_Params(c2,c3), ...
    @(prm,pf)   BurrCFAR_TLog(prm,pf), ...
    3, 'core', 'Two shape parameters; most expensive of the five in hardware');

d(end+1) = entry('GammaTex-MAP7', ...
    @(I,s,g,pf,varargin) GammaTexCFAR_Floating(I,s,g,pf,varargin{:}), ...
    [], [], ...
    2, 'core', 'K-family route: Gamma-MAP despeckle (own MapWindow=7, independent of sli) then plain-Gamma CFAR on the residual texture -- the two-stage architecture REF6/REF7 (Mahapatra et al.) actually publish, and the one that rescues single-look K''s near-total real-data support-condition collapse (FINDINGS.md F13). params/tlog are [] (monolithic, like CA-CFAR): this detector needs its OWN front end from the despeckled image, not the raw-image front end the sweep shares across the other five, so it cannot use the shared estimator-factoring path -- see GammaTexCFAR_Floating.m''s header.');

d(end+1) = entry('K', ...
    @(I,s,g,pf,varargin) KCFAR_Floating(I,s,g,pf,varargin{:}), ...
    @(c2,c3,sk) KCFAR_Params(c2), ...
    @(prm,pf)   KCFAR_TLog(prm,pf), ...
    2, 'variant', 'Single-look K-distribution applied directly to raw data (no despeckling) -- c2-only (1-D) shape like Weibull/Lognormal, needs a numerically-built delta grid (no elementary quantile) instead of a closed form. Superseded by GammaTex-MAP7 as the ''core'' K-family entry (2026-09-25): single-look K''s support condition collapses on real data (met by only ~6%/~0.05% of SSDD/HRSID windows at sli=17, FINDINGS.md F13), kept here as a documented negative-result variant, not removed.');

d(end+1) = entry('GenGamma-exact', ...
    @(I,s,g,pf,varargin) GenGammaCFAR_Floating(I,s,g,pf,'Solver','exact',varargin{:}), ...
    @(c2,c3,sk) GenGammaCFAR_Params(c2,c3,'Solver','exact'), ...
    @(prm,pf)   GenGammaCFAR_TLog(prm,pf), ...
    3, 'variant', 'Inverts the true MoLC relation instead of its cubic approximation');

d(end+1) = entry('G0-L1', ...
    @(I,s,g,pf,varargin) G0CFAR_Floating(I,s,g,pf,'Mode','L1','L',1,varargin{:}), ...
    @(c2,c3,sk) G0CFAR_Params(c2,c3,'Mode','L1','L',1), ...
    @(prm,pf)   G0CFAR_TLog(prm,pf), ...
    2, 'variant', 'Single-look: 1-D shape ROM, but fails wherever c2 < psi(1,1)/4');

d(end+1) = entry('CA-CFAR', ...
    @(I,s,g,pf,varargin) CACFAR_Floating(I,s,g,pf,varargin{:}), ...
    [], [], ...
    0, 'baseline', 'Textbook non-parametric comparator (BC-5): no shape fit, alpha=N*(Pfa^(-1/N)-1), assumes exponential intensity clutter');

switch want
    case 'all',      dets = d;
    case 'core',     dets = d(strcmp({d.group}, 'core'));
    case 'variant',  dets = d(strcmp({d.group}, 'variant'));
    case 'baseline', dets = d(strcmp({d.group}, 'baseline'));
end
end

function e = entry(name, fcn, params, tlog, nmom, group, note)
    e = struct('name', name, 'fcn', fcn, 'params', params, 'tlog', tlog, ...
               'nmom', nmom, 'group', group, 'note', note);
end
