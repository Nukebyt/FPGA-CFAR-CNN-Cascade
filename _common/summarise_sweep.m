function S = summarise_sweep(T)
%SUMMARISE_SWEEP  Pool over images within each (detector, sli, Pfa) cell.
%   Ship-level Pd is POOLED (total ships detected / total ships), not
%   averaged over images: a per-image mean over-weights images containing a
%   single ship. Pfa is likewise pooled over pixels.
%
%   Shared by run_comparison.m (SSDD) and run_comparison_hrsid.m (HRSID) so
%   pooling methodology is identical across datasets -- this is the exact
%   logic the SLI-plateau analysis reads, so any drift between the two
%   sweeps would silently corrupt a cross-dataset comparison.
    [g, det, sli, guard, pfa] = findgroups(T.Detector, T.Sli, T.Guard, T.Pfa);
    S = table(det, sli, guard, pfa, 'VariableNames', ...
        {'Detector','Sli','Guard','Pfa'});

    S.N                 = splitapply(@(x) x(1),        T.N,                g);
    S.NMoments          = splitapply(@(x) x(1),        T.NMoments,         g);
    S.Images            = splitapply(@numel,           T.Pd_ship,          g);
    S.ShipsTotal        = splitapply(@sum,             T.TotalShips,       g);
    S.ShipsDetected     = splitapply(@sum,             T.DetectedShips,    g);
    S.Pd_pooled         = S.ShipsDetected ./ S.ShipsTotal;
    S.FalsePixels       = splitapply(@sum,             T.FalsePixels,      g);
    S.BackgroundPixels  = splitapply(@sum,             T.BackgroundPixels, g);
    S.Pfa_pooled        = S.FalsePixels ./ S.BackgroundPixels;
    S.PfaRatio          = S.Pfa_pooled ./ S.Pfa;
    S.TP                = splitapply(@sum,             T.TP,               g);
    S.FP                = splitapply(@sum,             T.FP,               g);
    S.FN                = splitapply(@sum,             T.FN,               g);
    S.Precision_pooled  = S.TP ./ max(S.TP + S.FP, 1);
    S.Recall_pooled     = S.TP ./ max(S.TP + S.FN, 1);
    S.F1_pooled         = 2*S.Precision_pooled.*S.Recall_pooled ./ ...
                          max(S.Precision_pooled + S.Recall_pooled, eps);
    S.FractionInvalid   = splitapply(@mean,            T.FractionInvalid,  g);
    S.CoverageFraction  = splitapply(@mean,            T.CoverageFraction, g);
    S.MaxLogMismatchPct = splitapply(@max,             T.LogDomainMismatchPct, g);
    S.MedianShape       = splitapply(@(x) median(x,'omitnan'), T.MedianShape, g);
    S.IQRShape          = splitapply(@(x) median(x,'omitnan'), T.IQRShape,    g);
    S.MedianC2          = splitapply(@(x) median(x,'omitnan'), T.MedianC2,    g);
    S.MedianSkew        = splitapply(@(x) median(x,'omitnan'), T.MedianSkew,  g);

    % Throughput: the estimator alone, and the whole chain, per megapixel.
    S.EstMsPerMP   = splitapply(@(t,mp) sum(t)./sum(mp), T.TimeEstimatorMs, T.Megapixels, g);
    S.TotalMsPerMP = splitapply(@(a,b,c,mp) sum(a+b+c)./sum(mp), ...
        T.TimeFrontEndMs, T.TimeEstimatorMs, T.TimeThresholdMs, T.Megapixels, g);

    S = sortrows(S, {'Pfa','Sli','Detector'});
end
