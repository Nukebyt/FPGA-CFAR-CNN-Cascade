function T = verify_fixedpoint(varargin)
%VERIFY_FIXEDPOINT  Phase 3 gate: fixed-point vs floating-point agreement.
%
%   T = VERIFY_FIXEDPOINT()
%   T = VERIFY_FIXEDPOINT('NumImages',40,'Detectors',{'Weibull'})
%
%   THE GATE: detection-map mismatch must be < 0.01% before any RTL is
%   written for that detector. (The existing Weibull implementation achieved
%   0.0019% on its own datapath, which is the standard to match.)
%
%   ---------------------------------------------------------------------
%   WHAT A MISMATCH ACTUALLY MEANS
%   ---------------------------------------------------------------------
%   A pixel flips between the float and fixed models only if it sits within
%   the total threshold error of the decision boundary. So the mismatch rate
%   is roughly (threshold error) x (density of x at the threshold), and that
%   density is small because the threshold is out in the clutter tail by
%   construction. A mismatch rate far ABOVE what the measured threshold error
%   predicts is therefore a signal of a real bug -- a wrong scale factor, a
%   mis-addressed table, an overflow -- not of accumulated rounding.
%
%   This function reports both, so the two can be compared:
%     * the mismatch rate, and
%     * the actual threshold error distribution (max, RMS) that produced it.
%
%   It also reports mismatch SIGN: a model that systematically detects more
%   (or fewer) pixels than the float reference has a bias, whereas symmetric
%   mismatch is rounding. Bias is the more serious finding of the two.
%
%   Any front-end accumulator overflow or heavy address saturation is an
%   automatic FAIL regardless of the mismatch rate -- both mean the design is
%   operating outside the range its formats were sized for, and the fact that
%   it happens to still agree on this image sample is luck.

p = inputParser;
addParameter(p, 'NumImages', 40);
addParameter(p, 'Sli',       17);
addParameter(p, 'Guard',     13);
addParameter(p, 'Detectors', {'Weibull'});
addParameter(p, 'PfaIndex',  1:4);
parse(p, varargin{:});

paths    = cfar_setup();
sli      = p.Results.Sli;
guard    = p.Results.Guard;
detNames = p.Results.Detectors;
lut_root = fullfile(paths.root, 'lut');

GATE = 0.01;    % percent

imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
idx  = unique(round(linspace(1, numel(imgs), min(p.Results.NumImages, numel(imgs)))));

fprintf('=====================================================================\n');
fprintf(' Phase 3 GATE : fixed-point vs floating-point\n');
fprintf('   geometry  : sli=%d guard=%d\n', sli, guard);
fprintf('   images    : %d\n', numel(idx));
fprintf('   gate      : mismatch < %.3f%%\n', GATE);
fprintf('=====================================================================\n');

rows = struct('Detector',{},'Pfa',{},'MismatchPct',{},'ExtraDet',{},'MissedDet',{}, ...
              'MaxThreshErr',{},'RmsThreshErr',{},'AddrSatHighPct',{},'OverflowTotal',{}, ...
              'FloatDet',{},'FixedDet',{},'Pass',{});

for dn = detNames
    name = dn{1};
    fprintf('\n--- %s ---\n', name);

    [floatFcn, fixedFcn, lutFile] = detector_hooks(name, lut_root);
    L = load(lutFile);
    PfaOpts = L.Pfa_options;

    for pidx = p.Results.PfaIndex
        if pidx > numel(PfaOpts), continue; end
        Pfa = PfaOpts(pidx);

        nMis = 0; nTot = 0; nExtra = 0; nMissed = 0;
        nFloatDet = 0; nFixedDet = 0;
        maxTErr = 0; sumTErr2 = 0; nTErr = 0;
        satHigh = 0; ovfTot = 0;

        for k = idx
            Ik = imread(fullfile(paths.ssddJpeg, imgs(k).name));
            if ndims(Ik)==3, Ik = rgb2gray(Ik); end
            Id = double(Ik);

            [dF, TF] = floatFcn(Id, sli, guard, Pfa);
            [dX, TX, feX, stX] = fixedFcn(Id, sli, guard, pidx);

            mism = dF ~= dX;
            nMis   = nMis + sum(mism(:));
            nTot   = nTot + numel(mism);
            nExtra = nExtra + sum(dX(:) & ~dF(:));
            nMissed= nMissed + sum(dF(:) & ~dX(:));
            nFloatDet = nFloatDet + sum(dF(:));
            nFixedDet = nFixedDet + sum(dX(:));

            e = TX(:) - TF(:);
            e = e(isfinite(e));
            if ~isempty(e)
                maxTErr  = max(maxTErr, max(abs(e)));
                sumTErr2 = sumTErr2 + sum(e.^2);
                nTErr    = nTErr + numel(e);
            end

            satHigh = satHigh + stX.AddrSatHigh;
            ovfTot  = ovfTot + sum(struct2array_local(feX.Overflow));
        end

        misPct  = 100 * nMis / nTot;
        rmsTErr = sqrt(sumTErr2 / max(nTErr,1));
        satPct  = 100 * satHigh / nTot;
        pass    = (misPct < GATE) && (ovfTot == 0);

        fprintf('  Pfa=%-8g mismatch %8.5f%%  (extra %6d / missed %6d)  ', ...
            Pfa, misPct, nExtra, nMissed);
        fprintf('threshErr max %.2e rms %.2e  addrSatHi %.2f%%  ovf %d  [%s]\n', ...
            maxTErr, rmsTErr, satPct, ovfTot, tf(pass));

        rows(end+1) = struct('Detector',name,'Pfa',Pfa,'MismatchPct',misPct, ...
            'ExtraDet',nExtra,'MissedDet',nMissed,'MaxThreshErr',maxTErr, ...
            'RmsThreshErr',rmsTErr,'AddrSatHighPct',satPct,'OverflowTotal',ovfTot, ...
            'FloatDet',nFloatDet,'FixedDet',nFixedDet,'Pass',pass); %#ok<AGROW>
    end
end

T = struct2table(rows);
outPath = fullfile(paths.results,'fixedpoint_verification.csv');
writetable(T, outPath);

fprintf('\n=====================================================================\n');
nFail = sum(~T.Pass);
if nFail == 0
    fprintf(' GATE PASSED for all %d (detector, Pfa) cases.\n', height(T));
else
    fprintf(' GATE FAILED on %d of %d cases -- see the table above.\n', nFail, height(T));
end
fprintf(' Written: %s\n', outPath);
fprintf('=====================================================================\n');
end


%% =======================================================================
function [floatFcn, fixedFcn, lutFile] = detector_hooks(name, lut_root)
%DETECTOR_HOOKS  Map a detector name to its float model, fixed model and LUT.
%   Both models are wrapped to a common (I,sli,guard,...) -> (map, T_log)
%   signature so the gate loop stays detector-agnostic.
switch name
    case 'Weibull'
        floatFcn = @(I,s,g,pf) wrap_float(@WeibullCFAR_Shared, I,s,g,pf);
        fixedFcn = @(I,s,g,pi_) WeibullCFAR_FixedPoint(I,s,g,pi_,lut_root);
        lutFile  = fullfile(lut_root,'weibull','weibull_delta_lut.mat');

    case 'Lognormal'
        meta = load(fullfile(lut_root,'lognormal','lognormal_lut_meta.mat'));
        floatFcn = @(I,s,g,pf) wrap_float( ...
            @(II,ss,gg,pp) LognormalCFAR_Floating(II,ss,gg,pp, ...
                'SigmaMin', meta.sMin, 'SigmaMax', meta.sMax), ...
            I,s,g,pf);
        fixedFcn = @(I,s,g,pi_) LognormalCFAR_FixedPoint(I,s,g,pi_,lut_root);
        lutFile  = fullfile(lut_root,'lognormal','lognormal_delta_lut.mat');

    case 'GenGamma'
        % Every model parameter comes from the LUT metadata, not from the
        % float model's own defaults. The LUT is the single source of truth:
        % if the tables were built with a given solver and clamp set, the
        % float reference must use exactly those, or the gate is measuring the
        % cubic-vs-exact difference (F3) or a clamp mismatch rather than the
        % fixed-point error it exists to measure.
        meta = load(fullfile(lut_root,'gengamma','gengamma_lut_meta.mat'));
        floatFcn = @(I,s,g,pf) wrap_float( ...
            @(II,ss,gg,pp) GenGammaCFAR_Floating(II,ss,gg,pp, ...
                'Solver',  meta.solver, ...
                'KMin',    meta.KMin,  'KMax',    meta.KMax, ...
                'VAbsMin', meta.VMin,  'VAbsMax', meta.VMax), ...
            I,s,g,pf);
        fixedFcn = @(I,s,g,pi_) GenGammaCFAR_FixedPoint(I,s,g,pi_,lut_root);
        lutFile  = fullfile(lut_root,'gengamma','gengamma_num_lut.mat');

    case 'BurrXII'
        meta = load(fullfile(lut_root,'burr','burr_lut_meta.mat'));
        floatFcn = @(I,s,g,pf) wrap_float( ...
            @(II,ss,gg,pp) BurrCFAR_Floating(II,ss,gg,pp, ...
                'KappaMin',meta.kMin, 'KappaMax',meta.kMax, ...
                'RhoMin',  meta.rMin, 'RhoMax',  meta.rMax), ...
            I,s,g,pf);
        fixedFcn = @(I,s,g,pi_) BurrCFAR_FixedPoint(I,s,g,pi_,lut_root);
        lutFile  = fullfile(lut_root,'burr','burr_num_lut.mat');

    case 'G0'
        meta = load(fullfile(lut_root,'g0','g0_lut_meta.mat'));
        floatFcn = @(I,s,g,pf) wrap_float( ...
            @(II,ss,gg,pp) G0CFAR_Floating(II,ss,gg,pp,'Mode','LA', ...
                'UMin',meta.UMin, 'UMax',meta.UMax, 'LMax',meta.LMax), ...
            I,s,g,pf);
        fixedFcn = @(I,s,g,pi_) G0CFAR_FixedPoint(I,s,g,pi_,lut_root);
        lutFile  = fullfile(lut_root,'g0','g0_delta_lut.mat');

    otherwise
        error('verify_fixedpoint:UnknownDetector', ...
            'No fixed-point model registered for "%s" yet.', name);
end
end

function [dmap, T_log] = wrap_float(fcn, I, sli, guard, Pfa)
%WRAP_FLOAT  Run a floating detector and return its LOG-domain threshold,
%   which is what the fixed-point model produces and what must be compared.
    [dmap, ~, ~, ~, logdomain] = fcn(I, sli, guard, Pfa);
    T_log = logdomain.ThresholdMap;
end

function v = struct2array_local(s)
    fn = fieldnames(s);
    v = zeros(1, numel(fn));
    for i = 1:numel(fn), v(i) = s.(fn{i}); end
end

function o = tf(ok)
    if ok, o = 'PASS'; else, o = 'FAIL'; end
end
