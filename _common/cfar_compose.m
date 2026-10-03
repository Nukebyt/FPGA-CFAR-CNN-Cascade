function [detection_map, threshold_map, prm, stats, logdomain] = ...
    cfar_compose(I, sli, guard, Pfa, paramsFcn, tlogFcn, name, feOpts)
%CFAR_COMPOSE  Assemble a complete CFAR detector from its two distribution-
%   specific pieces plus the shared front end.
%
%   Every <Dist>CFAR_Floating.m in this project is a thin wrapper around this
%   function. Centralising the composition guarantees that the five detectors
%   differ ONLY in their estimator and their threshold offset -- identical
%   padding, identical windowing, identical decision rule, identical
%   invalid-window handling, identical statistics. That is what makes the
%   Phase 2 comparison a fair one.
%
%   INPUTS
%     I, sli, guard, Pfa : as for any of the detectors
%     paramsFcn : @(c2, c3, skew) -> prm struct  (must set .valid)
%     tlogFcn   : @(prm, Pfa)     -> delta = T_log - c1
%     name      : detector name, for messages
%     feOpts    : cell array of name-value options forwarded to
%                 cfar_front_end, OR a precomputed front-end struct from a
%                 previous cfar_front_end call (the sweep reuses one front
%                 end across all detectors and all Pfa values -- the front
%                 end depends on neither).
%
%   ---------------------------------------------------------------------
%   THE DECISION, AND THE TWO DOMAINS
%   ---------------------------------------------------------------------
%   Log domain (what the hardware does):
%       x_cut > T_log,          T_log = c1 + delta
%   Amplitude/intensity domain (the textbook form):
%       I_cut > T_int,          T_int = exp(2*T_log) - 0.5
%
%   These are algebraically the same test, since x = log(sqrt(I+0.5)) is
%   strictly increasing: x > T_log <=> sqrt(I+0.5) > exp(T_log)
%   <=> I > exp(2*T_log) - 0.5. BOTH are computed on every call and their
%   agreement is reported in `logdomain`, so the equivalence is checked on
%   real data rather than trusted -- the same discipline the Weibull model
%   used, and the reason the -0.5 term is present (omitting it, as the
%   original scripts do, leaves a half-LSB band where the two domains
%   genuinely disagree on integer-valued imagery).
%
%   ---------------------------------------------------------------------
%   INVALID WINDOWS
%   ---------------------------------------------------------------------
%   Three of the five estimators have genuine out-of-support conditions
%   (Generalized Gamma's r <= r_max, G0's 4*c2 > psi(1,1), Burr's
%   s in (-1.1394, 2)). Where prm.valid is false the detector CANNOT form a
%   threshold, and this function declares NO DETECTION there -- matching what
%   the original CFAR Generalized Gamma/legacy script does on its
%   3*c3^2 <= 8*c2^3 branch.
%
%   This is reported, never hidden: stats.FractionInvalid is a first-class
%   Phase 2 metric. A detector that silently declares "no target" on a third
%   of its windows is not achieving a low false-alarm rate, it is failing to
%   run, and the two look identical if you only plot Pfa.
%
%   OUTPUTS  (identical contract to WeibullCFAR_Floating.m)
%     detection_map : HxW logical, the log-domain decision (canonical)
%     threshold_map : HxW intensity-domain threshold T_int
%     prm           : the estimator's parameter struct, with .c1/.c2/.c3
%                     maps attached for LUT range analysis
%     stats         : summary struct
%     logdomain     : .ThresholdMap, .DetectionMap (amplitude-domain),
%                     .NumMismatch, .PercentMismatch

if nargin < 8 || isempty(feOpts)
    feOpts = {};
end

if isstruct(feOpts)
    fe = feOpts;
    if fe.sli ~= sli || fe.guard ~= guard
        error('cfar_compose:FrontEndMismatch', ...
            'Supplied front end was built for sli=%d/guard=%d, not %d/%d.', ...
            fe.sli, fe.guard, sli, guard);
    end
else
    fe = cfar_front_end(I, sli, guard, feOpts{:});
end

%% ---- Distribution-specific estimator -----------------------------------
prm = paramsFcn(fe.c2, fe.c3, fe.skew);
if ~isfield(prm, 'valid')
    error('cfar_compose:NoValidField', '%s params function must set .valid', name);
end

%% ---- Distribution-specific threshold offset -----------------------------
delta = tlogFcn(prm, Pfa);

%% ---- Log-domain decision (canonical) -----------------------------------
T_log = fe.c1 + delta;
valid = prm.valid & isfinite(T_log);
detection_map = (fe.x > T_log) & valid;

%% ---- Amplitude/intensity-domain cross-check ----------------------------
T_int = exp(2 * T_log) - 0.5;
detect_amp = (double(I) > T_int) & valid;

mism = detect_amp ~= detection_map;
logdomain = struct();
logdomain.ThresholdMap    = T_log;
logdomain.DetectionMap    = detect_amp;
logdomain.NumMismatch     = sum(mism(:));
logdomain.PercentMismatch = 100 * logdomain.NumMismatch / numel(mism);

threshold_map = T_int;

%% ---- Attach front-end maps for Phase 3 LUT range analysis --------------
prm.c1   = fe.c1;
prm.c2   = fe.c2;
prm.c3   = fe.c3;
prm.skew = fe.skew;
prm.x    = fe.x;

%% ---- Stats --------------------------------------------------------------
stats = struct();
stats.Detector           = name;
stats.NumReferenceCells  = fe.N;
stats.sli                = sli;
stats.guard              = guard;
stats.Pfa                = Pfa;
stats.NumDetections      = sum(detection_map(:));
stats.FractionInvalid    = 1 - mean(valid(:));
stats.NumValid           = sum(valid(:));
stats.MeanThresholdLog   = mean(T_log(valid), 'omitnan');
stats.LogDomainMismatchPct = logdomain.PercentMismatch;
if isfield(prm, 'FractionAtMin'), stats.FractionAtMin = prm.FractionAtMin; else, stats.FractionAtMin = NaN; end
if isfield(prm, 'FractionAtMax'), stats.FractionAtMax = prm.FractionAtMax; else, stats.FractionAtMax = NaN; end
end
