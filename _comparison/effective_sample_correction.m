function effective_sample_correction()
%EFFECTIVE_SAMPLE_CORRECTION  Effective-independent-window Clopper-Pearson
%   correction for spatially correlated reference windows (hostile-review
%   items #2 and #10, 2026-09-25).
%
%   compute_clopper_pearson.m's intervals treat every background PIXEL as an
%   independent Bernoulli trial. They are not: reference windows of size
%   sli x sli overlap heavily as the cell-under-test slides across an image,
%   so the number of genuinely independent windows in a sweep cell is much
%   smaller than the raw pixel count. This script uses this project's own
%   window geometry to estimate that correction directly, rather than
%   leaving it as an unquantified "somewhat wider" caveat (as the paper's
%   §4 originally did):
%
%       N_eff = BackgroundPixels / sli^2
%       k_eff = round(FalsePixels / sli^2)
%
%   (dividing both the trial count and the success count by the same
%   sli^2 factor keeps the point estimate k_eff/N_eff close to the
%   original FalsePixels/BackgroundPixels ratio, while shrinking both to a
%   scale representing genuinely-spaced-apart windows -- an approximate,
%   stated-as-such correction, not an exact spatial-covariance model).
%
%   RESULT (the reason this script exists): applying this correction to the
%   paper's own §3 reconciliation shows the two GenGamma SSDD sli=21,
%   Pfa=1e-6 measurements (3,162.1x from a 100-image draw, 3,027.6x from an
%   independent 60-image draw) are NOT actually in tension once spatial
%   correlation is accounted for -- the 100-image figure's naive i.i.d. CI
%   is [3,133.9x, 3,190.5x] (1.8% relative width, does NOT contain 3,027.6x),
%   but its effective-N-corrected CI is roughly [2,608x, 3,829x] (38.5%
%   relative width), which comfortably contains 3,027.6x. This is exactly
%   what §4's own independence caveat predicted before this script existed
%   to quantify it.
%
%   Reproduce: cfar_setup(); effective_sample_correction();

paths = cfar_setup();
alpha = 0.05;

cells = { ...
    'comparison_summary_main_ci.csv',              'GenGamma', 21,  1e-6, '100-image headline (F9/original)'; ...
    'comparison_summary_ssdd_full_fixed60_ci.csv',  'GenGamma', 21,  1e-6, '60-image reconciliation point'; ...
    'comparison_summary_ssdd_full_fixed60_ci.csv',  'GenGamma', 151, 1e-6, 'SSDD 60-image ceiling'; ...
    'comparison_summary_ssdd_full_fixed60_ci.csv',  'Weibull',  51,  1e-6, 'Weibull, SSDD'; ...
    'comparison_summary_hrsid_full_ci.csv',         'Weibull',  51,  1e-6, 'Weibull, HRSID'; ...
    'comparison_summary_hrsid_full_ci.csv',         'Lognormal',51,  1e-6, 'Lognormal, HRSID'};

fprintf('=====================================================================\n');
fprintf(' Effective-independent-window Clopper-Pearson correction\n');
fprintf('=====================================================================\n');
fprintf('%-40s %8s %10s %12s %22s\n', 'Cell', 'Sli', 'Ratio', 'Naive CI', 'Effective-N CI (rel. width)');

for i = 1:size(cells,1)
    T = readtable(fullfile(paths.results, cells{i,1}));
    mask = strcmp(T.Detector, cells{i,2}) & T.Sli == cells{i,3} & abs(T.Pfa - cells{i,4}) < cells{i,4}*1e-6;
    r = T(mask, :);
    if height(r) == 0
        fprintf('  [not found] %s\n', cells{i,5});
        continue;
    end
    r = r(1,:);
    sli2 = cells{i,3}^2;
    Neff = r.BackgroundPixels / sli2;
    keff = round(r.FalsePixels / sli2);

    if keff <= 0
        lo = 0;
    else
        lo = betaincinv(alpha/2, keff, Neff - keff + 1);
    end
    if keff >= Neff
        hi = 1;
    else
        hi = betaincinv(1 - alpha/2, keff + 1, Neff - keff);
    end
    ratioEff = (keff / Neff) / r.Pfa;
    loR = lo / r.Pfa; hiR = hi / r.Pfa;
    relw = 100 * (hiR - loR) / ratioEff;

    fprintf('%-40s %8d %9.1fx  [%7.1fx,%8.1fx] -> [%8.1fx,%9.1fx] (%.1f%%)\n', ...
        cells{i,5}, cells{i,3}, r.PfaRatio, r.Ratio_CI_Lo, r.Ratio_CI_Hi, loR, hiR, relw);
end

%% The specific reconciliation check the paper's §3/§6.2 needs -------------
fprintf('\n--- Reconciliation check: does the 60-image point fall inside the\n');
fprintf('    100-image headline''s EFFECTIVE-N interval? ---\n');
T1 = readtable(fullfile(paths.results, 'comparison_summary_main_ci.csv'));
r1 = T1(strcmp(T1.Detector,'GenGamma') & T1.Sli==21 & abs(T1.Pfa-1e-6)<1e-12, :);
r1 = r1(1,:);
sli2 = 21^2;
Neff = r1.BackgroundPixels / sli2;
keff = round(r1.FalsePixels / sli2);
lo = betaincinv(alpha/2, keff, Neff-keff+1);
hi = betaincinv(1-alpha/2, keff+1, Neff-keff);
loR = lo / r1.Pfa; hiR = hi / r1.Pfa;

T2 = readtable(fullfile(paths.results, 'comparison_summary_ssdd_full_fixed60_ci.csv'));
r2 = T2(strcmp(T2.Detector,'GenGamma') & T2.Sli==21 & abs(T2.Pfa-1e-6)<1e-12, :);
r2 = r2(1,:);

inside = (r2.PfaRatio >= loR) && (r2.PfaRatio <= hiR);
fprintf('  100-image effective-N 95%% CI: [%.1fx, %.1fx]\n', loR, hiR);
fprintf('  60-image point estimate:      %.1fx\n', r2.PfaRatio);
fprintf('  Inside interval: %d\n', inside);

end
