function compute_clopper_pearson()
%COMPUTE_CLOPPER_PEARSON  Attach exact (Clopper-Pearson) 95% binomial
%   confidence intervals to every achieved-Pfa figure this project reports
%   (Paper 1, roadmap item 4).
%
%   Computed from the CLUTTER-WINDOW count (`BackgroundPixels`, order
%   1e5-1e7 per sweep cell), NOT the ship count (order 100-900) -- the
%   roadmap's own explicit instruction, since conflating the two is "the
%   single most obvious small-sample objection a TGRS reviewer will raise
%   against a Pfa-measurement paper."
%
%   For each row of the existing comparison_summary_*.csv files (already in
%   the repo, no new sweep needed -- FalsePixels/BackgroundPixels are
%   already pooled per (detector, sli, Pfa) cell), treats FalsePixels as
%   Binomial(BackgroundPixels, true_Pfa) successes and computes the EXACT
%   Clopper-Pearson interval via the incomplete-beta identity:
%
%       lower = betaincinv(alpha/2,   k,   n-k+1)      (0 if k=0)
%       upper = betaincinv(1-alpha/2, k+1, n-k)        (1 if k=n)
%
%   with k=FalsePixels, n=BackgroundPixels, alpha=0.05 (95% CI). This is
%   the textbook exact interval (inverts the binomial CDF directly via its
%   incomplete-beta relation), not a normal approximation -- appropriate
%   here since several cells have small k (e.g. GenGamma's own worst-case
%   divergence cells still have k in the thousands-to-tens-of-thousands,
%   but other detectors/geometries run k in the tens).
%
%   CAVEAT stated explicitly, not hidden: this treats each background pixel
%   as an independent Bernoulli trial. Neighbouring reference-window pixels
%   are NOT independent (they share overlapping windows and spatially
%   correlated clutter) -- so the true interval is somewhat wider than the
%   i.i.d. Clopper-Pearson interval computed here. This is the same
%   approximation implicit in treating `BackgroundPixels` as the sample
%   size at all (rather than an "effective" smaller count); flagged here so
%   the paper states it as a limitation rather than an implicit assumption.
%
%   OUTPUT: comparison_summary_<tag>_ci.csv next to each input, with two
%   new column pairs: Pfa_CI_Lo/Pfa_CI_Hi (achieved-Pfa 95% CI) and
%   Ratio_CI_Lo/Ratio_CI_Hi (= Pfa_CI/nominal Pfa, since nominal Pfa is a
%   fixed constant, not a random variable -- the ratio's CI is just the
%   Pfa CI rescaled).

paths = cfar_setup();
alpha = 0.05;

files = { fullfile(paths.results, 'comparison_summary_main.csv'), ...
          fullfile(paths.results, 'comparison_summary_ssdd_full_fixed60.csv'), ...
          fullfile(paths.results, 'comparison_summary_hrsid_full.csv') };

for fi = 1:numel(files)
    f = files{fi};
    if ~isfile(f)
        fprintf('SKIP (not found): %s\n', f);
        continue;
    end
    T = readtable(f);
    n = height(T);
    loP = nan(n,1); hiP = nan(n,1); loR = nan(n,1); hiR = nan(n,1); width = nan(n,1);
    for i = 1:n
        k = T.FalsePixels(i);
        N = T.BackgroundPixels(i);
        if N <= 0
            continue;
        end
        if k <= 0
            lo = 0;
        else
            lo = betaincinv(alpha/2, k, N-k+1);
        end
        if k >= N
            hi = 1;
        else
            hi = betaincinv(1-alpha/2, k+1, N-k);
        end
        loP(i) = lo; hiP(i) = hi;
        loR(i) = lo / T.Pfa(i); hiR(i) = hi / T.Pfa(i);
        width(i) = hi - lo;
    end
    T.Pfa_CI_Lo   = loP;   T.Pfa_CI_Hi   = hiP;
    T.Ratio_CI_Lo = loR;   T.Ratio_CI_Hi = hiR;
    T.CI_Width    = width;

    [~, name, ~] = fileparts(f);
    outPath = fullfile(paths.results, [name '_ci.csv']);
    writetable(T, outPath);
    fprintf('Wrote %s (%d rows)\n', outPath, n);
end

%% Headline figures, printed directly for the paper -----------------------
fprintf('\n=====================================================================\n');
fprintf(' Headline divergence figures with 95%% Clopper-Pearson intervals\n');
fprintf('=====================================================================\n');

headline_row('comparison_summary_main_ci.csv', 'GenGamma', 21, 1e-6, ...
    'Original F9 headline (100 images, sli=21)');
headline_row('comparison_summary_ssdd_full_fixed60_ci.csv', 'GenGamma', 151, 1e-6, ...
    'SSDD 60-image reconciliation ceiling (sli=151)');
headline_row('comparison_summary_ssdd_full_fixed60_ci.csv', 'Weibull', 51, 1e-6, ...
    'Weibull, matched geometry, SSDD');
headline_row('comparison_summary_hrsid_full_ci.csv', 'Weibull', 51, 1e-6, ...
    'Weibull, matched geometry, HRSID');
headline_row('comparison_summary_hrsid_full_ci.csv', 'Lognormal', 51, 1e-6, ...
    'Lognormal, matched geometry, HRSID');
end

function headline_row(fname, detector, sli, pfa, label)
    paths = cfar_setup();
    T = readtable(fullfile(paths.results, fname));
    mask = strcmp(T.Detector, detector) & T.Sli == sli & abs(T.Pfa - pfa) < pfa*1e-6;
    if ~any(mask)
        fprintf('  [not found] %s %s sli=%d Pfa=%g\n', label, detector, sli, pfa);
        return;
    end
    r = T(mask, :);
    r = r(1,:);
    fprintf('\n  %s\n', label);
    fprintf('    %s, sli=%d, nominal Pfa=%g, N_background=%d, FalsePixels=%d\n', ...
        detector, sli, pfa, r.BackgroundPixels, r.FalsePixels);
    fprintf('    PfaRatio = %.1fx   95%% CI = [%.1fx, %.1fx]   (relative width %.1f%%)\n', ...
        r.PfaRatio, r.Ratio_CI_Lo, r.Ratio_CI_Hi, ...
        100*(r.Ratio_CI_Hi - r.Ratio_CI_Lo)/r.PfaRatio);
end
