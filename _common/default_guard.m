function guard = default_guard(sli)
%DEFAULT_GUARD  Best-guard-for-this-sli, from the existing Weibull project's
%   own parameter sweep (README section 2.7) where that sweep covered the
%   size, and log-interpolated onto the 0.65-0.80 guard/sli band elsewhere.
%   Always returned odd and strictly less than sli, as cfar_front_end requires.
%
%   Shared by run_comparison.m (SSDD) and run_comparison_hrsid.m (HRSID) so
%   both datasets use identical window geometry for a given Sli.
    known = containers.Map( ...
        {11, 15, 17, 21, 31, 41, 51, 61, 71, 81, 91}, ...
        { 7, 11, 13, 15, 21, 27, 41, 49, 57, 65, 73});
    guard = zeros(size(sli));
    for i = 1:numel(sli)
        if isKey(known, sli(i))
            guard(i) = known(sli(i));
        else
            g = round(0.72 * sli(i));
            if mod(g,2) == 0, g = g - 1; end
            guard(i) = max(1, min(g, sli(i)-2));
        end
    end
end
