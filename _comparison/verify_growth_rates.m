function verify_growth_rates()
%VERIFY_GROWTH_RATES  Cross-check the Pfa->0 asymptotic growth-rate
%   derivations for GenGamma/G0/BurrXII/K (Paper 1, roadmap item 2) against
%   the actual *_TLog implementations -- exactly the discipline used for
%   the pre-existing Weibull/Lognormal check (F10, 6.7e-16 agreement): a
%   derivation that doesn't match the real code is a bug, not a footnote.
%
%   Two things get checked per detector:
%     (a) Absolute error between the exact TLog delta and the derived
%         asymptotic formula, at the SAME Pfa range this project's sweeps
%         actually use (1e-1 ... 1e-6), at shape values spanning what the
%         real data sweeps report (see MedianShape/IQRShape columns in
%         comparison_summary_*.csv).
%     (b) The same comparison pushed to much smaller Pfa (down to 1e-40),
%         confirming the error -> 0 (i.e. this really is the leading
%         asymptotic term, not a coincidence at moderate Pfa).
%
%   Burr XII's formula is EXACT for all Pfa (its survival function is
%   algebraically invertible) -- included here for completeness, and its
%   error should be at numerical-precision level even at Pfa=1e-1.
%
%   See PAPER1_DRAFT_reachability-bound.md section 2 for the derivations
%   and PAPER1_ROADMAP's item 2 for why this check is required before the
%   growth rates go in the paper.

EULER_GAMMA = 0.5772156649015329;
PfaGrid = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-9 1e-12 1e-20 1e-30 1e-40];

fprintf('=====================================================================\n');
fprintf(' Growth-rate derivation cross-check (Paper 1, item 2)\n');
fprintf('=====================================================================\n\n');

%% ---- Generalized Gamma ------------------------------------------------
fprintf('--- Generalized Gamma ---\n');
fprintf('  v>0 branch: delta ~ [loglog(1/Pfa) - psi(k)] / v\n');
fprintf('  v<0 branch: delta ~ [log(1/Pfa)/k + log(gamma(k+1))/k - psi(k)] / v\n\n');

kv_pairs_pos = [1.0 0.5; 1.0 1.5; 2.0 1.0; 5.0 0.8];   % v > 0
kv_pairs_neg = [1.0 -0.5; 1.0 -1.5; 2.0 -1.0; 5.0 -0.8]; % v < 0

for kv = [kv_pairs_pos; kv_pairs_neg]'
    k = kv(1); v = kv(2);
    prm = struct('k', k, 'v', v, 'valid', true);
    fprintf('  k=%.2f v=%+.2f\n', k, v);
    for Pfa = PfaGrid
        actual = GenGammaCFAR_TLog(prm, Pfa);
        L = -log(Pfa);
        if v > 0
            pred = (log(L) - psi(k)) / v;
        else
            % log G ~ -(1/k)*L + (1/k)*log(gamma(k+1))  (G -> 0 as Pfa -> 0,
            % so log G -> -inf; sign was wrong in an earlier draft of this
            % check -- caught by this exact cross-check, see BUG_LOG).
            pred = (-L/k + log(gamma(k+1))/k - psi(k)) / v;
        end
        fprintf('    Pfa=%9.1e  actual=%10.5f  pred=%10.5f  abs.err=%.3e\n', ...
            Pfa, actual, pred, abs(actual - pred));
    end
end

%% ---- G0 (L1 mode: L fixed at typical single-look values) --------------
fprintf('\n--- G0 ---\n');
fprintf('  delta ~ 0.5*(psi(u)-psi(L)) + [log(1/Pfa) - log(u*B(u,L))] / (2u)\n\n');

uL_pairs = [1.0 1.0; 3.0 1.0; 8.0 1.0; 3.0 3.0];
for uL = uL_pairs'
    u = uL(1); L = uL(2);
    prm = struct('u', u, 'L', L, 'valid', true);
    fprintf('  u=%.2f L=%.2f\n', u, L);
    for Pfa = PfaGrid
        actual = G0CFAR_TLog(prm, Pfa);
        Lp = -log(Pfa);
        logUB = log(u) + betaln(u, L);
        pred = 0.5*(psi(u) - psi(L)) + (Lp - logUB) / (2*u);
        fprintf('    Pfa=%9.1e  actual=%10.5f  pred=%10.5f  abs.err=%.3e\n', ...
            Pfa, actual, pred, abs(actual - pred));
    end
end

%% ---- Burr XII (exact, not merely asymptotic) ---------------------------
fprintf('\n--- Burr XII ---\n');
fprintf('  delta = [psi(kappa)-psi(1) + log(1/Pfa)/kappa] / rho   (EXACT to O(exp(-w)))\n\n');

kr_pairs = [1.0 1.0; 3.0 1.0; 6.0 2.0; 1.0 4.0];
for kr = kr_pairs'
    kappa = kr(1); rho = kr(2);
    prm = struct('kappa', kappa, 'rho', rho, 'valid', true);
    fprintf('  kappa=%.2f rho=%.2f\n', kappa, rho);
    for Pfa = PfaGrid
        actual = BurrCFAR_TLog(prm, Pfa);
        Lp = -log(Pfa);
        pred = (psi(kappa) - psi(1) + Lp/kappa) / rho;
        fprintf('    Pfa=%9.1e  actual=%10.5f  pred=%10.5f  abs.err=%.3e\n', ...
            Pfa, actual, pred, abs(actual - pred));
    end
end

%% ---- K-distribution -----------------------------------------------------
fprintf('\n--- K-distribution ---\n');
fprintf('  delta ~ loglog(1/Pfa) - 0.5*log(4a) - kappa1(a),  kappa1(a)=0.5*(psi(a)-log(a)-gammaE)\n\n');

% smaller extreme-Pfa grid here: the fzero+integral solve in KCFAR_TLog is
% not verified for arbitrarily extreme Pfa the way the elementary functions
% above are (gammaincinv/betaincinv are library-verified at extreme args;
% the mixture integral is only smoke-tested to Pfa=1e-6 in this project).
PfaGridK = [1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-9 1e-12 1e-15 1e-20];
a_vals = [0.5 1.5 3.0 6.0];
for a = a_vals
    prm = struct('a', a, 'AMin', 0.01, 'AMax', 1e4, 'valid', true);
    fprintf('  a=%.2f\n', a);
    for Pfa = PfaGridK
        actual = KCFAR_TLog(prm, Pfa);
        L = -log(Pfa);
        kappa1 = 0.5 * (psi(a) - log(a) - EULER_GAMMA);
        pred = log(L) - 0.5*log(4*a) - kappa1;
        fprintf('    Pfa=%9.1e  actual=%10.5f  pred=%10.5f  abs.err=%.3e\n', ...
            Pfa, actual, pred, abs(actual - pred));
    end
end

fprintf('\n=====================================================================\n');
fprintf(' Done. Expect: Burr XII abs.err at numerical-precision level for ALL\n');
fprintf(' Pfa (exact formula); GenGamma/G0/K abs.err shrinking toward 0 as Pfa\n');
fprintf(' shrinks (asymptotic formulas -- large error at Pfa=1e-1 is expected\n');
fprintf(' and NOT a bug, only convergence as Pfa->0 matters).\n');
fprintf('=====================================================================\n');
end
