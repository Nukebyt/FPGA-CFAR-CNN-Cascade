function [medShape, iqrShape] = shape_summary(prm)
%SHAPE_SUMMARY  Median and IQR of the detector's PRIMARY shape parameter.
%   Reported because the spread of the shape estimate across windows is a
%   direct proxy for how finely the Phase 3 ROM has to resolve it: a shape
%   parameter that barely moves needs few entries, one that swings across
%   decades needs a log-addressed table.
    v = prm.valid;
    if ~any(v(:)), medShape = NaN; iqrShape = NaN; return; end
    switch prm.Name
        case 'Weibull',           s = prm.C(v);
        case 'Lognormal',         s = prm.sigma(v);
        case 'GeneralizedGamma',  s = prm.k(v);
        case 'G0',                s = prm.u(v);
        case 'BurrXII',           s = prm.kappa(v);
        otherwise,                s = NaN;
    end
    s = s(isfinite(s));
    if isempty(s), medShape = NaN; iqrShape = NaN; return; end
    medShape = median(s);
    q = prctile(s, [25 75]);
    iqrShape = q(2) - q(1);
end
