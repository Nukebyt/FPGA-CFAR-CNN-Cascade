function print_comparison_headline(S, PfaList)
%PRINT_COMPARISON_HEADLINE  The table worth reading straight out of the sweep.
    Pf = PfaList(min(2, numel(PfaList)));     % 1e-4 by default
    fprintf('\n--- Headline: Pfa = %g, pooled over all images ---\n', Pf);
    fprintf('%-16s %5s %6s %8s %10s %9s %8s %8s %9s\n', ...
        'Detector','Sli','N','Pd','Pfa_meas','Pfa/nom','F1','invalid%','est ms/MP');
    sub = S(S.Pfa == Pf, :);
    sub = sortrows(sub, {'Sli','Detector'});
    for i = 1:height(sub)
        fprintf('%-16s %5d %6d %8.3f %10.2e %9.1f %8.3f %8.1f %9.1f\n', ...
            sub.Detector{i}, sub.Sli(i), sub.N(i), sub.Pd_pooled(i), ...
            sub.Pfa_pooled(i), sub.PfaRatio(i), sub.F1_pooled(i), ...
            100*sub.FractionInvalid(i), sub.EstMsPerMP(i));
    end
end
