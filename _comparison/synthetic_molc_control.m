function synthetic_molc_control(Ntrials)
%SYNTHETIC_MOLC_CONTROL  Exactly-specified-model Monte-Carlo control for the
%   Pfa reachability-bound paper (hostile-review finding #1, 2026-09-25:
%   every real-data divergence figure in this paper comes from data that is
%   simultaneously (a) subject to finite-window MoLC estimation noise and
%   (b) not exactly described by the fitted family -- so the paper cannot
%   yet show how much of its measured divergence is the growth-rate
%   mechanism it derives in Section 2 versus ordinary model mismatch).
%
%   This script removes (b) entirely. For each of the six clutter families,
%   synthetic log-amplitude data is drawn EXACTLY from that family, from a
%   KNOWN true shape parameter (no fitting error possible in the generating
%   process, by construction) -- then the SAME finite-window MoLC estimator
%   this project's real detectors use is applied to a window of the SAME
%   size the real sweeps use, and the resulting fitted threshold is tested
%   against an independent draw from the SAME true distribution. Any
%   divergence from the nominal Pfa that remains is attributable ONLY to
%   (a) finite-window estimation noise interacting with (c) the delta(shape,
%   Pfa) growth-rate mechanism Section 2 derives -- exactly the two effects
%   the paper's thesis says should be sufficient, with model mismatch
%   controlled out.
%
%   DESIGN
%   ------
%   Per trial: draw N = sli^2 - guard^2 i.i.d. log-amplitude samples from
%   the true distribution (the "reference window"), compute their unbiased
%   sample log-cumulants (c1,c2,c3) via the identical formula
%   cfar_front_end.m uses, fit shape via the real *_Params.m, form the
%   threshold via the real *_TLog.m at each nominal Pfa, then draw ONE
%   further independent sample from the SAME true distribution (the "CUT
%   pixel") and test it against the threshold. Pooling this Bernoulli
%   exceedance test over many independent trials gives a measured Pfa with
%   an exact Clopper-Pearson interval, by the same construction as
%   compute_clopper_pearson.m.
%
%   sli/guard pairs are taken directly from default_guard.m (the same
%   geometry the real HRSID/SSDD sweeps use), so N above is the real
%   reference-window sample count, not an arbitrary "large" fit sample --
%   this is what distinguishes this control from verify_models.m's own
%   Section 5 calibration check, which fits from an unrealistically large
%   4e6-sample pool per condition (adequate to validate that the FORMULA is
%   implemented correctly, but not to expose realistic finite-window
%   estimation noise).
%
%   THREE SHAPE REGIMES, none invented for this script:
%     'SSDD-representative'  : true (c2,c3) = this project's own real median
%                               (c2, skew) at sli=21/Pfa=1e-6, pooled directly
%                               from comparison_summary_ssdd_full_fixed60.csv,
%                               fed through each detector's own *_Params.m.
%     'HRSID-representative' : same, from comparison_summary_hrsid_full.csv.
%     'Moderate-established' : the exact demo shape values verify_models.m's
%                               own Section 5 calibration check already uses
%                               (Weibull C=2.0, Lognormal sigma=0.6, GenGamma
%                               k=1.6/v=1.3, G0 L=2.0/u=4.0, Burr
%                               kappa=2.2/rho=1.8, K a=2.0) -- reused
%                               unchanged, not re-picked for this script.
%   The first two derive every family's true shape from ONE shared raw
%   (c2,c3) point, which is not guaranteed to fall inside every family's
%   support region -- and, consistent with this project's own F6 finding,
%   frequently does not for G0/Burr XII at SSDD's real heterogeneity level.
%   Where a regime's (c2,c3) point is outside a family's support, that
%   family has no valid true distribution to draw from under that regime at
%   all; this is detected up front and the (regime, family) cell is skipped
%   and reported as inapplicable, not silently coerced to a mangled result.
%   The third regime exists precisely so every family has at least one
%   common ground for direct comparison.
%
%   Reproduce: cfar_setup(); synthetic_molc_control();
%   Output:    Results/synthetic_molc_control.csv

if nargin < 1 || isempty(Ntrials)
    Ntrials = 300000;
end

paths = cfar_setup();
rng(20260925, 'twister');

sliList   = [21 51 151];
guardList = default_guard(sliList);
PfaList   = [1e-2 1e-3 1e-4];
MaxCells  = 1e7;              % memory cap per generation batch (elements)

detNames = {'Weibull', 'Lognormal', 'GenGamma', 'G0', 'BurrXII', 'K'};

regimes = struct('name', {}, 'mode', {}, 'c2', {}, 'skew', {}, 'shapes', {});
regimes(1) = struct('name', 'SSDD-representative', 'mode', 'moments', ...
    'c2', 0.116847898667876, 'skew', -1.64140194953972, 'shapes', []);
regimes(2) = struct('name', 'HRSID-representative', 'mode', 'moments', ...
    'c2', 0.108167498172672, 'skew', -1.0200093248858, 'shapes', []);
regimes(3) = struct('name', 'Moderate-established', 'mode', 'direct', ...
    'c2', [], 'skew', [], 'shapes', struct( ...
        'Weibull',   struct('C', 2.0), ...
        'Lognormal', struct('sigma', 0.6), ...
        'GenGamma',  struct('k', 1.6, 'v', 1.3), ...
        'G0',        struct('L', 2.0, 'u', 4.0), ...
        'BurrXII',   struct('kappa', 2.2, 'rho', 1.8), ...
        'K',         struct('a', 2.0)));

rows = struct('Regime', {}, 'Sli', {}, 'N', {}, 'Detector', {}, 'TrueShape', {}, ...
    'Pfa', {}, 'Trials', {}, 'ValidTrials', {}, 'FractionInvalid', {}, ...
    'Exceed', {}, 'MeasuredPfa', {}, 'PfaRatio', {}, 'RatioCI_Lo', {}, 'RatioCI_Hi', {});

fprintf('=====================================================================\n');
fprintf(' Synthetic exactly-specified-model MoLC control\n');
fprintf(' (hostile-review item #1 -- isolates finite-window estimation noise\n');
fprintf('  + growth-rate mechanism from real-data model mismatch)\n');
fprintf('=====================================================================\n');

for r = 1:numel(regimes)
    reg = regimes(r);
    trueP = struct();
    applicable = struct();

    if strcmp(reg.mode, 'moments')
        c2t = reg.c2;
        c3t = reg.skew * c2t^1.5;
        trueP.Weibull   = WeibullCFAR_Params(c2t);
        trueP.Lognormal = LognormalCFAR_Params(c2t);
        trueP.GenGamma  = GenGammaCFAR_Params(c2t, c3t, 'Solver', 'exact');
        trueP.G0        = G0CFAR_Params(c2t, c3t, 'Mode', 'LA', 'Iters', 60);
        trueP.BurrXII   = BurrCFAR_Params(c2t, c3t);
        trueP.K         = KCFAR_Params(c2t);
        fprintf('\n--- Regime: %s  (shared true c2=%.4f, skew=%.4f) ---\n', reg.name, c2t, reg.skew);
    else
        s = reg.shapes;
        trueP.Weibull   = struct('C', s.Weibull.C, 'valid', true);
        trueP.Lognormal = struct('sigma', s.Lognormal.sigma, 'valid', true);
        trueP.GenGamma  = struct('k', s.GenGamma.k, 'v', s.GenGamma.v, 'valid', true);
        trueP.G0        = struct('L', s.G0.L, 'u', s.G0.u, 'valid', true);
        trueP.BurrXII   = struct('kappa', s.BurrXII.kappa, 'rho', s.BurrXII.rho, 'valid', true);
        trueP.K         = struct('a', s.K.a, 'valid', true);
        fprintf('\n--- Regime: %s (per-family established shapes, verify_models.m Section 5 values) ---\n', reg.name);
    end

    applicable.Weibull   = isfinite(trueP.Weibull.C)   && trueP.Weibull.C > 0;
    applicable.Lognormal = isfinite(trueP.Lognormal.sigma) && trueP.Lognormal.sigma > 0;
    applicable.GenGamma  = isfinite(trueP.GenGamma.k)  && isfinite(trueP.GenGamma.v) && trueP.GenGamma.k > 0;
    applicable.G0        = isfinite(trueP.G0.L) && isfinite(trueP.G0.u) && trueP.G0.L > 0 && trueP.G0.u > 0;
    applicable.BurrXII   = isfinite(trueP.BurrXII.kappa) && isfinite(trueP.BurrXII.rho) && trueP.BurrXII.kappa > 0;
    applicable.K         = isfinite(trueP.K.a) && trueP.K.a > 0;

    for d = 1:numel(detNames)
        nm = detNames{d};
        if applicable.(nm)
            fprintf('  %-10s true shape: %s\n', nm, shape_str(nm, trueP));
        else
            fprintf('  %-10s NOT APPLICABLE under this regime -- shared (c2,c3) point falls outside this family''s support (no valid true distribution exists)\n', nm);
        end
    end

    for si = 1:numel(sliList)
        sli = sliList(si); guard = guardList(si);
        N = sli^2 - guard^2;
        batchSize = max(200, floor(MaxCells / N));
        nBatches = ceil(Ntrials / batchSize);

        exceed   = zeros(numel(detNames), numel(PfaList));
        total    = zeros(numel(detNames), 1);
        nInvalid = zeros(numel(detNames), 1);

        for b = 1:nBatches
            n = min(batchSize, Ntrials - (b-1)*batchSize);
            if n <= 0, break; end

            %% Weibull: A = (-log U)^(1/C)
            if applicable.Weibull
                Cw = trueP.Weibull.C;
                xw = (1/Cw) .* log(-log(rand(n, N)));
                [c1h, c2h, ~] = row_molc(xw);
                testx = (1/Cw) .* log(-log(rand(n, 1)));
                do_detector(1, c2h, [], c1h, testx, PfaList, @(c2,c3) WeibullCFAR_Params(c2), @WeibullCFAR_TLog);
            end

            %% Lognormal: A = exp(sigma*Z)
            if applicable.Lognormal
                sg = trueP.Lognormal.sigma;
                xl = sg .* randn(n, N);
                [c1h, c2h, ~] = row_molc(xl);
                testx = sg .* randn(n, 1);
                do_detector(2, c2h, [], c1h, testx, PfaList, @(c2,c3) LognormalCFAR_Params(c2), @LognormalCFAR_TLog);
            end

            %% GenGamma: A = (G/k)^(1/v), G ~ Gamma(k,1)
            if applicable.GenGamma
                kk = trueP.GenGamma.k; vv = trueP.GenGamma.v;
                Gg = randg_local(kk, n, N);
                xg = (1/vv) .* log(Gg ./ kk);
                [c1h, c2h, c3h] = row_molc(xg);
                Gt = randg_local(kk, n, 1);
                testx = (1/vv) .* log(Gt ./ kk);
                do_detector(3, c2h, c3h, c1h, testx, PfaList, ...
                    @(c2,c3) GenGammaCFAR_Params(c2, c3, 'Solver', 'exact'), @GenGammaCFAR_TLog);
            end

            %% G0: Z = (Gamma(L,1)/L) / (Gamma(u,1)/u), A = sqrt(Z)
            if applicable.G0
                LL = trueP.G0.L; uu = trueP.G0.u;
                num = randg_local(LL, n, N) / LL;
                den = randg_local(uu, n, N) / uu;
                x0 = 0.5 .* log(num ./ den);
                [c1h, c2h, c3h] = row_molc(x0);
                numT = randg_local(LL, n, 1) / LL;
                denT = randg_local(uu, n, 1) / uu;
                testx = 0.5 .* log(numT ./ denT);
                do_detector(4, c2h, c3h, c1h, testx, PfaList, ...
                    @(c2,c3) G0CFAR_Params(c2, c3, 'Mode', 'LA', 'Iters', 60), @G0CFAR_TLog);
            end

            %% Burr XII: A = (U^(-1/kappa) - 1)^(1/rho)
            if applicable.BurrXII
                kap = trueP.BurrXII.kappa; rh = trueP.BurrXII.rho;
                xb = (1/rh) .* log(rand(n, N).^(-1/kap) - 1);
                [c1h, c2h, c3h] = row_molc(xb);
                testx = (1/rh) .* log(rand(n, 1).^(-1/kap) - 1);
                do_detector(5, c2h, c3h, c1h, testx, PfaList, ...
                    @(c2,c3) BurrCFAR_Params(c2, c3), @BurrCFAR_TLog);
            end

            %% K (single-look): tau ~ Gamma(a,1)/a [mean 1], I = Exp(tau), A = sqrt(I)
            if applicable.K
                ak = trueP.K.a;
                tauK = randg_local(ak, n, N) / ak;
                Ik = -tauK .* log(rand(n, N));
                xk = 0.5 .* log(Ik);
                [c1h, c2h, ~] = row_molc(xk);
                tauT = randg_local(ak, n, 1) / ak;
                IkT = -tauT .* log(rand(n, 1));
                testx = 0.5 .* log(IkT);
                do_detector(6, c2h, [], c1h, testx, PfaList, @(c2,c3) KCFAR_Params(c2), @KCFAR_TLog);
            end
        end

        for d = 1:numel(detNames)
            nm = detNames{d};
            if ~applicable.(nm)
                continue;
            end
            tshape = shape_str(nm, trueP);
            for pi = 1:numel(PfaList)
                Pf = PfaList(pi);
                k = exceed(d, pi);
                Ntot = total(d);
                if Ntot <= 0
                    continue;
                end
                measured = k / Ntot;
                ratio = measured / Pf;
                if k <= 0
                    lo = 0;
                else
                    lo = betaincinv(0.025, k, Ntot - k + 1);
                end
                if k >= Ntot
                    hi = 1;
                else
                    hi = betaincinv(0.975, k + 1, Ntot - k);
                end
                row = struct('Regime', reg.name, 'Sli', sli, 'N', N, ...
                    'Detector', nm, 'TrueShape', tshape, 'Pfa', Pf, ...
                    'Trials', Ntrials, 'ValidTrials', Ntot, ...
                    'FractionInvalid', nInvalid(d) / Ntrials, ...
                    'Exceed', k, 'MeasuredPfa', measured, 'PfaRatio', ratio, ...
                    'RatioCI_Lo', lo / Pf, 'RatioCI_Hi', hi / Pf);
                rows(end+1) = row; %#ok<AGROW>
            end
            if total(d) > 0
                fprintf('  sli=%3d  %-10s (true %-28s) FracInvalid=%5.1f%%  ', ...
                    sli, nm, tshape, 100*nInvalid(d)/Ntrials);
                for pi = 1:numel(PfaList)
                    fprintf('Pfa=%g: ratio=%6.2fx  ', PfaList(pi), exceed(d,pi)/total(d)/PfaList(pi));
                end
                fprintf('\n');
            end
        end
    end
end

T = struct2table(rows);
outPath = fullfile(paths.results, 'synthetic_molc_control.csv');
writetable(T, outPath);
fprintf('\nWrote %s (%d rows)\n', outPath, height(T));

    %% ---- nested helper: fit + threshold + accumulate exceedance counts ----
    function do_detector(idx, c2h, c3h, c1h, testx, PfaL, paramFcn, tlogFcn)
        if isempty(c3h)
            prm = paramFcn(c2h, []);
        else
            prm = paramFcn(c2h, c3h);
        end
        valid = prm.valid;
        nInvalid(idx) = nInvalid(idx) + sum(~valid);
        total(idx) = total(idx) + sum(valid);
        for ppi = 1:numel(PfaL)
            delta = tlogFcn(prm, PfaL(ppi));
            Tlog = c1h + delta;
            hit = valid & (testx > Tlog);
            exceed(idx, ppi) = exceed(idx, ppi) + sum(hit);
        end
    end

end

%% ===========================================================================
function [c1, c2, c3] = row_molc(x)
%ROW_MOLC  Per-row (per-trial) unbiased sample log-cumulants over a
%   trials-by-N matrix of log-amplitude samples -- identical formula to
%   cfar_front_end.m's central-moment-to-k-statistic conversion, applied
%   row-wise instead of over an image window.
    N = size(x, 2);
    S1 = sum(x, 2);
    S2 = sum(x.^2, 2);
    S3 = sum(x.^3, 2);
    m1 = S1 / N;
    m2 = S2 / N - m1.^2;
    m3 = S3 / N - 3 * m1 .* (S2 / N) + 2 * m1.^3;
    c1 = m1;
    c2 = max((N / (N - 1)) * m2, 1e-9);
    c3 = (N^2 / ((N - 1) * (N - 2))) * m3;
end

function s = shape_str(name, trueP)
    switch name
        case 'Weibull',   s = sprintf('C=%.3f', trueP.Weibull.C);
        case 'Lognormal', s = sprintf('sigma=%.3f', trueP.Lognormal.sigma);
        case 'GenGamma',  s = sprintf('k=%.3f,v=%.3f', trueP.GenGamma.k, trueP.GenGamma.v);
        case 'G0',        s = sprintf('L=%.3f,u=%.3f', trueP.G0.L, trueP.G0.u);
        case 'BurrXII',   s = sprintf('kappa=%.3f,rho=%.3f', trueP.BurrXII.kappa, trueP.BurrXII.rho);
        case 'K',         s = sprintf('a=%.3f', trueP.K.a);
    end
end
