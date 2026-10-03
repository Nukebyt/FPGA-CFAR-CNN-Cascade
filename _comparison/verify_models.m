%% verify_models.m
%
% Phase 1 gate. Runs every check that must pass before the Phase 2 sweep is
% worth running at all. Nothing here is a formality -- each test targets a
% specific way one of these models could be silently wrong.
%
%   1. Shared front end vs. the hardware-validated Weibull model.
%      WeibullCFAR_Shared (built on cfar_front_end) must match
%      WeibullCFAR_Floating (carried over unmodified from the existing
%      project, verified bit-exact against its RTL and validated on a real
%      DE10-Standard) to floating-point rounding. If cfar_front_end ever
%      drifts from the windowing/padding the RTL implements, it shows up
%      here and nowhere else.
%
%   2. Log-domain == amplitude-domain, for all five detectors.
%      Every detector computes both decisions on every call. They are
%      algebraically identical, so any mismatch is a bug in a derivation.
%
%   3. Estimator round-trips.
%      Each estimator is fed log-cumulants generated FROM KNOWN parameters
%      and must recover them. This is the only test that can catch a wrong
%      MoLC inversion -- comparing detector outputs to each other cannot,
%      since all five would be wrong together.
%
%   4. Support conditions behave as derived.
%
%   5. Threshold calibration on synthetic clutter.
%      Each detector is run on synthetic clutter drawn from ITS OWN
%      distribution, where the measured false-alarm rate must land near the
%      nominal Pfa. A detector that fails this has a threshold formula error
%      that no amount of comparison against the other four would reveal.

clc; close all;
paths = cfar_setup();

fprintf('=====================================================================\n');
fprintf(' Phase 1 verification\n');
fprintf('=====================================================================\n');

nfail = 0;

%% =====================================================================
%% 1. Shared front end vs. hardware-validated Weibull model
%% =====================================================================
fprintf('\n--- 1. Shared front end vs. WeibullCFAR_Floating ---\n');

imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
Itest = double(rgb2gray_safe(imread(fullfile(paths.ssddJpeg, imgs(1).name))));

for cfg = [15 11; 21 15; 31 21]'
    sli = cfg(1); guard = cfg(2);
    [dA, tA, ~, sA] = WeibullCFAR_Floating(Itest, sli, guard, 1e-3);
    [dB, tB, ~, sB] = WeibullCFAR_Shared  (Itest, sli, guard, 1e-3);

    dmap = sum(dA(:) ~= dB(:));
    % Compare thresholds only where both are finite and in a sane range;
    % exp(2*T_log) can overflow on pathological windows in either model.
    m = isfinite(tA) & isfinite(tB) & tA < 1e12 & tB < 1e12;
    tdiff = max(abs(tA(m) - tB(m)) ./ max(abs(tA(m)), 1));

    ok = (dmap == 0) && (tdiff < 1e-9);
    nfail = nfail + ~ok;
    fprintf('  sli=%2d guard=%2d : detection-map diff %d px, max rel threshold diff %.3e  [%s]\n', ...
        sli, guard, dmap, tdiff, tf(ok));
    fprintf('                    N: %d vs %d, detections: %d vs %d\n', ...
        sA.NumReferenceCells, sB.NumReferenceCells, sA.NumDetections, sB.NumDetections);
end

%% =====================================================================
%% 2. Log-domain == amplitude-domain, all five detectors
%% =====================================================================
fprintf('\n--- 2. Log-domain vs amplitude-domain decision (all detectors) ---\n');

sli = 21; guard = 15; Pfa = 1e-3;
fe = cfar_front_end(Itest, sli, guard);

dets = detector_registry();
for k = 1:numel(dets)
    [~, ~, ~, st, ld] = dets(k).fcn(Itest, sli, guard, Pfa, 'FrontEnd', fe);
    ok = ld.PercentMismatch < 1e-4;
    nfail = nfail + ~ok;
    fprintf('  %-18s mismatch %6d px (%.6f%%), invalid %6.2f%%, detections %6d  [%s]\n', ...
        dets(k).name, ld.NumMismatch, ld.PercentMismatch, ...
        100*st.FractionInvalid, st.NumDetections, tf(ok));
end

%% =====================================================================
%% 3. Estimator round-trips from known parameters
%% =====================================================================
fprintf('\n--- 3. Estimator round-trips (synthetic log-cumulants) ---\n');

EG = 0.5772156649015329;

% ---- Weibull: c2 = psi(1,1)/C^2 -------------------------------------
Ctrue = [0.9 1.5 2.0 3.5 7.0];
prm = WeibullCFAR_Params(pi^2/6 ./ Ctrue.^2);
e = max(abs(prm.C - Ctrue) ./ Ctrue);
report('Weibull  C', e, 1e-12); nfail = nfail + (e >= 1e-12);

% ---- Lognormal: c2 = sigma^2 ----------------------------------------
sig = [0.1 0.3 0.7 1.5];
prm = LognormalCFAR_Params(sig.^2);
e = max(abs(prm.sigma - sig) ./ sig);
report('Lognormal sigma', e, 1e-12); nfail = nfail + (e >= 1e-12);

% ---- Generalized Gamma: c2 = psi(1,k)/v^2, c3 = psi(2,k)/v^3 ---------
% Tested against the 'exact' solver, which inverts the true MoLC relation.
% ('cubic' reproduces the supplied formula, which is an asymptotic
%  approximation and is NOT expected to round-trip -- its error against the
%  exact relation is quantified separately in step 3b.)
ktrue = [0.3 0.8 1.5 4.0 12.0];
vtrue = [1.2 -0.9 2.5 -1.7 0.6];
c2s = psi(1, ktrue) ./ vtrue.^2;
c3s = psi(2, ktrue) ./ vtrue.^3;
prm = GenGammaCFAR_Params(c2s, c3s, 'Solver', 'exact');
ek = max(abs(prm.k - ktrue) ./ ktrue);
ev = max(abs(prm.v - vtrue) ./ abs(vtrue));
report('GenGamma k (exact)', ek, 1e-8); nfail = nfail + (ek >= 1e-8);
report('GenGamma v (exact)', ev, 1e-8); nfail = nfail + (ev >= 1e-8);

fprintf('  3b. cubic vs exact solver on the same (c2,c3):\n');
prmc = GenGammaCFAR_Params(c2s, c3s, 'Solver', 'cubic');
for i = 1:numel(ktrue)
    fprintf('      k_true=%6.2f -> exact %8.4f, cubic %8.4f (%+7.2f%%)\n', ...
        ktrue(i), prm.k(i), prmc.k(i), 100*(prmc.k(i)-ktrue(i))/ktrue(i));
end

% ---- G0: psi(1,L)+psi(1,u) = 4c2,  psi(2,L)-psi(2,u) = 8c3 -----------
Ltrue = [1.0 2.0 4.0 1.5];
utrue = [3.0 6.0 1.8 12.0];
c2s = (psi(1,Ltrue) + psi(1,utrue)) / 4;
c3s = (psi(2,Ltrue) - psi(2,utrue)) / 8;
prm = G0CFAR_Params(c2s, c3s, 'Mode', 'LA', 'Iters', 60);
eL = max(abs(prm.L - Ltrue) ./ Ltrue);
eu = max(abs(prm.u - utrue) ./ utrue);
report('G0 L  (LA mode)', eL, 1e-6); nfail = nfail + (eL >= 1e-6);
report('G0 u  (LA mode)', eu, 1e-6); nfail = nfail + (eu >= 1e-6);

% ---- G0 single-look: L fixed at 1 ------------------------------------
utrue = [2.0 5.0 0.7];
c2s = (psi(1,1) + psi(1,utrue)) / 4;
prm = G0CFAR_Params(c2s, zeros(size(c2s)), 'Mode', 'L1', 'L', 1);
eu = max(abs(prm.u - utrue) ./ utrue);
report('G0 u  (L1 mode)', eu, 1e-10); nfail = nfail + (eu >= 1e-10);

% ---- Burr XII: c2 = (psi(1,k)+psi(1,1))/rho^2, c3 = (psi(2,1)-psi(2,k))/rho^3
ktrue = [0.4 1.0 2.5 8.0];
rtrue = [1.1 2.0 0.7 3.3];
c2s = (psi(1,ktrue) + psi(1,1)) ./ rtrue.^2;
c3s = (psi(2,1) - psi(2,ktrue)) ./ rtrue.^3;
prm = BurrCFAR_Params(c2s, c3s);
ek = max(abs(prm.kappa - ktrue) ./ ktrue);
er = max(abs(prm.rho - rtrue) ./ rtrue);
report('Burr kappa', ek, 1e-7); nfail = nfail + (ek >= 1e-7);
report('Burr rho',   er, 1e-7); nfail = nfail + (er >= 1e-7);

% ---- K (single-look, L=1): psi(1,a) = 4c2 - psi(1,1) -----------------
atrue = [0.3 1.0 4.0 15.0 50.0];
c2s = (psi(1,atrue) + psi(1,1)) / 4;
prm = KCFAR_Params(c2s);
ea = max(abs(prm.a - atrue) ./ atrue);
report('K a', ea, 1e-6); nfail = nfail + (ea >= 1e-6);

%% =====================================================================
%% 4. Support conditions
%% =====================================================================
fprintf('\n--- 4. Support conditions ---\n');

% G0 single-look: needs c2 > psi(1,1)/4 = 0.41123  (equivalently Weibull C < 2)
c2edge = psi(1,1)/4;
p1 = G0CFAR_Params(c2edge*0.99, 0, 'Mode','L1');
p2 = G0CFAR_Params(c2edge*1.01, 0, 'Mode','L1');
ok = ~p1.valid && p2.valid;
nfail = nfail + ~ok;
fprintf('  G0 L1 support c2 > %.5f : below=%d above=%d  [%s]\n', ...
    c2edge, p1.valid, p2.valid, tf(ok));

% Burr: needs -1.139443 < s < 2
p1 = BurrCFAR_Params(1, -1.20);       % s = -1.20, below the lower limit
p2 = BurrCFAR_Params(1, -1.00);       % s = -1.00, inside
p3 = BurrCFAR_Params(1,  2.50);       % s =  2.50, above the upper limit
ok = ~p1.valid && p2.valid && ~p3.valid;
nfail = nfail + ~ok;
fprintf('  Burr support -1.1394 < s < 2 : [%d %d %d] (want [0 1 0])  [%s]\n', ...
    p1.valid, p2.valid, p3.valid, tf(ok));

% Generalized gamma: cubic solver invertible for r < 8, exact for r < 4
p1 = GenGammaCFAR_Params(1,  sqrt(3.0), 'Solver','cubic');   % r = 3.0
p2 = GenGammaCFAR_Params(1,  sqrt(9.0), 'Solver','cubic');   % r = 9.0
p3 = GenGammaCFAR_Params(1,  sqrt(3.0), 'Solver','exact');   % r = 3.0
p4 = GenGammaCFAR_Params(1,  sqrt(5.0), 'Solver','exact');   % r = 5.0
ok = p1.valid && ~p2.valid && p3.valid && ~p4.valid;
nfail = nfail + ~ok;
fprintf('  GenGamma support cubic r<8 / exact r<4 : [%d %d %d %d] (want [1 0 1 0])  [%s]\n', ...
    p1.valid, p2.valid, p3.valid, p4.valid, tf(ok));

% Burr threshold must not overflow at small kappa / small Pfa -- the legacy
% Pfa^(-1/kappa) form returns Inf here.
prm_small = struct('kappa', 0.01, 'rho', 1, 'valid', true);
d = BurrCFAR_TLog(prm_small, 1e-6);
naive = (1e-6)^(-1/0.01) - 1;
ok = isfinite(d);
nfail = nfail + ~ok;
fprintf('  Burr overflow guard at kappa=0.01, Pfa=1e-6 : delta=%.4f (naive Pfa^(-1/k) = %g)  [%s]\n', ...
    d, naive, tf(ok));

% K: needs c2 > psi(1,1)/4 = 0.41123 -- numerically the SAME boundary as
% G0's L1 support above, not a coincidence: both reduce to "psi(1,other
% shape) = 4c2 - psi(1,1) must be a finite positive value" once L=1 is fixed.
c2edge_k = psi(1,1)/4;
pk1 = KCFAR_Params(c2edge_k*0.99);
pk2 = KCFAR_Params(c2edge_k*1.01);
ok = ~pk1.valid && pk2.valid;
nfail = nfail + ~ok;
fprintf('  K support c2 > %.5f : below=%d above=%d  [%s]\n', ...
    c2edge_k, pk1.valid, pk2.valid, tf(ok));

%% =====================================================================
%% 5. Threshold calibration on synthetic clutter
%% =====================================================================
% Each detector is run on a large synthetic clutter field drawn from its OWN
% distribution, with the parameters estimated from the data exactly as in the
% real detector. The measured exceedance rate must land near the nominal Pfa.
% This is the only check that validates the THRESHOLD formula (as opposed to
% the estimator) -- an error there would shift every detector's operating
% point without ever producing an internal inconsistency.
fprintf('\n--- 5. Threshold calibration on synthetic clutter (nominal vs measured Pfa) ---\n');
fprintf('    %-14s %10s %12s %12s %8s\n', 'detector', 'nominal', 'measured', 'ratio', 'n');

rng(12345);
Nsamp = 4e6;
PfaList = [1e-2 1e-3 1e-4];

for k = 1:numel(PfaList)
    Pf = PfaList(k);

    % ---- Weibull: A = B*(-log U)^(1/C) -----------------------------
    C = 2.0; B = 1.7;
    A = B .* (-log(rand(Nsamp,1))).^(1/C);
    calib('Weibull', A, Pf, @(c1,c2,c3) WeibullCFAR_TLog(WeibullCFAR_Params(c2), Pf));

    % ---- Lognormal --------------------------------------------------
    mu = 0.4; sg = 0.6;
    A = exp(mu + sg*randn(Nsamp,1));
    calib('Lognormal', A, Pf, @(c1,c2,c3) LognormalCFAR_TLog(LognormalCFAR_Params(c2), Pf));

    % ---- Generalized gamma: A = sigma*(G/k)^(1/v), G ~ Gamma(k,1) ----
    kk = 1.6; vv = 1.3; sg2 = 1.1;
    G = randg_local(kk, Nsamp, 1);
    A = sg2 .* (G./kk).^(1/vv);
    calib('GenGamma', A, Pf, @(c1,c2,c3) GenGammaCFAR_TLog( ...
        GenGammaCFAR_Params(c2, c3, 'Solver','exact'), Pf));

    % ---- G0: Z = (gam/u)*(Gamma(L,1)/L)/(Gamma(u,1)/u), A = sqrt(Z) --
    LL = 2.0; uu = 4.0; gam = 3.0;
    num = randg_local(LL, Nsamp, 1) / LL;
    den = randg_local(uu, Nsamp, 1) / uu;
    A = sqrt( (gam/uu) .* num ./ den );
    calib('G0', A, Pf, @(c1,c2,c3) G0CFAR_TLog( ...
        G0CFAR_Params(c2, c3, 'Mode','LA','Iters',60), Pf));

    % ---- Burr XII: A = eta*((U^(-1/kappa))-1)^(1/rho) ----------------
    kap = 2.2; rh = 1.8; eta = 1.4;
    U = rand(Nsamp,1);
    A = eta .* (U.^(-1/kap) - 1).^(1/rh);
    calib('BurrXII', A, Pf, @(c1,c2,c3) BurrCFAR_TLog(BurrCFAR_Params(c2, c3), Pf));

    % ---- K (single-look): tau~Gamma(a,1)/a [mean 1], I=Exp(tau), A=sqrt(I) -
    a_k = 2.0;
    tauK = randg_local(a_k, Nsamp, 1) / a_k;
    Ik = -tauK .* log(rand(Nsamp,1));
    A = sqrt(Ik);
    calib('K', A, Pf, @(c1,c2,c3) KCFAR_TLog(KCFAR_Params(c2), Pf));

    fprintf('\n');
end

%% =====================================================================
fprintf('=====================================================================\n');
if nfail == 0
    fprintf(' ALL STRUCTURAL CHECKS PASSED -- Phase 2 sweep can proceed.\n');
else
    fprintf(' %d CHECK(S) FAILED -- fix before running the Phase 2 sweep.\n', nfail);
end
fprintf(' (Section 5 is reported, not pass/failed: sampling noise at Pfa=1e-4\n');
fprintf('  over %g samples is itself ~%.0f%%, so read it as an order-of-magnitude\n', ...
    Nsamp, 100/sqrt(Nsamp*1e-4));
fprintf('  calibration check, not a precise one.)\n');
fprintf('=====================================================================\n');


%% ---------------------------------------------------------------------
function calib(name, A, Pf, deltaFcn)
%CALIB  Estimate parameters from a synthetic clutter sample drawn from the
%   detector's own distribution, form the threshold, and measure the actual
%   exceedance rate against the nominal Pfa.
    x  = log(A);
    n  = numel(x);
    c1 = mean(x);
    d  = x - c1;
    c2 = sum(d.^2) / (n-1);
    c3 = (n/((n-1)*(n-2))) * sum(d.^3);

    delta = deltaFcn(c1, c2, c3);
    if ~isfinite(delta)
        fprintf('    %-14s %10.0e %12s %12s %8d\n', name, Pf, 'INVALID', '--', n);
        return;
    end
    T = c1 + delta;
    measured = mean(x > T);
    fprintf('    %-14s %10.0e %12.3e %12.2f %8.0e\n', name, Pf, measured, measured/Pf, n);
end

function report(label, err, tol)
    fprintf('  %-20s max rel err %.3e  (tol %.0e)  [%s]\n', label, err, tol, tf(err < tol));
end

function s = tf(ok)
    if ok, s = 'PASS'; else, s = 'FAIL'; end
end

function g = rgb2gray_safe(im)
    if ndims(im) == 3, g = rgb2gray(im); else, g = im; end
end
