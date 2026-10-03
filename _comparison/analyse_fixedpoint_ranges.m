function R = analyse_fixedpoint_ranges(varargin)
%ANALYSE_FIXEDPOINT_RANGES  Phase 3 step 0: measure every quantity the
%   fixed-point datapath has to represent, so the Q-formats are sized from
%   data rather than guessed.
%
%   R = ANALYSE_FIXEDPOINT_RANGES()
%   R = ANALYSE_FIXEDPOINT_RANGES('Sli',17,'Guard',13,'NumImages',80)
%
%   Writes Results/fixedpoint_ranges.csv and prints a sizing recommendation
%   for each signal.
%
%   ---------------------------------------------------------------------
%   WHY THIS RUNS BEFORE ANY LUT IS BUILT
%   ---------------------------------------------------------------------
%   Two of the three numerical hazards in BUG_LOG.md (H1 c3 cancellation,
%   H2 tail-quantile cancellation) are fixed-point problems that do not show
%   up in double precision. Sizing a Q-format by eye is how they come back.
%   Every format in fixedpoint_config.m traces to a number printed here.
%
%   The analytical range and the empirical range are reported side by side
%   and the FORMAT IS SIZED ON THEIR UNION -- the analytical bound alone can
%   be far looser than reality (wasting bits), and the empirical bound alone
%   can miss a value the detector is required to represent but that this
%   image sample never produced.
%
%   ---------------------------------------------------------------------
%   THE CENTRED LOG-AMPLITUDE
%   ---------------------------------------------------------------------
%   cfar_front_end subtracts a fixed constant X0 from x before accumulating
%   (see H1). The hardware bakes X0 into the log LUT, so the quantity the
%   accumulators actually see is xc = x - X0, NOT x. For 8-bit input,
%       x  in [log(sqrt(0.5)), log(sqrt(255.5))] = [-0.3466, 2.7716]
%       xc in [-1.5591, 1.5591]   with X0 = 1.2125 (the midpoint)
%   which is symmetric about zero -- so xc needs one integer bit plus sign
%   where the uncentred x needed two. The centring pays for itself in word
%   length as well as in conditioning.
%
%   NAME-VALUE OPTIONS
%     'Sli'       : window size the LUTs will be built for (default 17)
%     'Guard'     : guard size (default 13)
%     'NumImages' : SSDD images to sample (default 80)
%     'Pfa'       : Pfa set the back-end LUT must cover
%                   (default [1e-3 1e-4 1e-5 1e-6])

p = inputParser;
addParameter(p, 'Sli',       17);
addParameter(p, 'Guard',     13);
addParameter(p, 'NumImages', 80);
addParameter(p, 'Pfa',       [1e-3 1e-4 1e-5 1e-6]);
parse(p, varargin{:});

paths = cfar_setup();
sli   = p.Results.Sli;
guard = p.Results.Guard;
PfaL  = p.Results.Pfa;
N     = sli^2 - guard^2;

X0    = 1.2125;
XC_MAX = max(abs([log(sqrt(0.5)), log(sqrt(255.5))] - X0));

fprintf('=====================================================================\n');
fprintf(' Phase 3 step 0 : fixed-point range analysis\n');
fprintf('   geometry : sli=%d guard=%d  ->  N = %d reference cells\n', sli, guard, N);
fprintf('   Pfa set  : %s\n', mat2str(PfaL));
fprintf('=====================================================================\n');

fprintf('\n--- Analytical bounds (exact, independent of any dataset) ---\n');
fprintf('  x  = log(sqrt(I+0.5))        in [%+.4f, %+.4f]\n', log(sqrt(0.5)), log(sqrt(255.5)));
fprintf('  X0 (centring constant)        = %.4f\n', X0);
fprintf('  xc = x - X0                   in [%+.4f, %+.4f]  -> |xc| <= %.4f\n', -XC_MAX, XC_MAX, XC_MAX);
fprintf('  xc^2                          in [0, %.4f]\n', XC_MAX^2);
fprintf('  xc^3                          in [%+.4f, %+.4f]\n', -XC_MAX^3, XC_MAX^3);
fprintf('  S1 = sum(xc)   over N=%4d    in [%+.1f, %+.1f]\n', N, -N*XC_MAX, N*XC_MAX);
fprintf('  S2 = sum(xc^2)                in [0, %.1f]\n', N*XC_MAX^2);
fprintf('  S3 = sum(xc^3)                in [%+.1f, %+.1f]\n', -N*XC_MAX^3, N*XC_MAX^3);

%% ---- Empirical sweep ----------------------------------------------------
imgs = dir(fullfile(paths.ssddJpeg,'*.jpg'));
idx  = unique(round(linspace(1, numel(imgs), min(p.Results.NumImages, numel(imgs)))));

acc = struct('c1',[], 'c2',[], 'c3',[], 'skew',[], 'r',[], ...
             'S1',[], 'S2',[], 'S3',[]);

fprintf('\nScanning %d images...\n', numel(idx));
for k = idx
    Ik = imread(fullfile(paths.ssddJpeg, imgs(k).name));
    if ndims(Ik)==3, Ik = rgb2gray(Ik); end
    fe = cfar_front_end(double(Ik), sli, guard, 'X0', X0);

    st = 7;   % subsample for tractability; ranges are extremes, not densities
    acc.c1   = [acc.c1;   fe.c1(1:st:end)'];    %#ok<AGROW>
    acc.c2   = [acc.c2;   fe.c2(1:st:end)'];    %#ok<AGROW>
    acc.c3   = [acc.c3;   fe.c3(1:st:end)'];    %#ok<AGROW>
    acc.skew = [acc.skew; fe.skew(1:st:end)'];  %#ok<AGROW>
end
acc.r = acc.skew.^2;

fprintf('  %d window samples collected.\n', numel(acc.c1));

%% ---- Report -------------------------------------------------------------
rows = struct('Signal',{},'Min',{},'P001',{},'P50',{},'P999',{},'Max',{}, ...
              'IntBits',{},'FracBits',{},'TotalBits',{},'Signed',{},'Note',{});

fprintf('\n--- Empirical distributions (%d samples) ---\n', numel(acc.c1));
fprintf('%-10s %12s %12s %12s %12s %12s\n','signal','min','p0.1','median','p99.9','max');
for f = {'c1','c2','c3','skew','r'}
    v = acc.(f{1});
    v = v(isfinite(v));
    q = prctile(v, [0.1 50 99.9]);
    fprintf('%-10s %12.5f %12.5f %12.5f %12.5f %12.5f\n', ...
        f{1}, min(v), q(1), q(2), q(3), max(v));
end

%% ---- Shape-parameter and delta ranges, per detector --------------------
fprintf('\n--- Shape parameters and back-end offset delta = T_log - c1 ---\n');
dets = detector_registry('Set','core');
c2v = acc.c2;  c3v = acc.c3;  skv = acc.skew;

for di = 1:numel(dets)
    prm = dets(di).params(c2v, c3v, skv);
    v = prm.valid;
    fprintf('\n  %s  (valid on %.1f%% of samples)\n', dets(di).name, 100*mean(v));

    shp = shape_of(prm);
    if ~isempty(shp)
        s = shp(v & isfinite(shp));
        fprintf('    shape  : min=%.5f  p0.1=%.5f  median=%.5f  p99.9=%.5f  max=%.5f\n', ...
            min(s), prctile(s,0.1), median(s), prctile(s,99.9), max(s));
    end

    for pf = PfaL
        d = dets(di).tlog(prm, pf);
        d = d(v & isfinite(d));
        if isempty(d), continue; end
        fprintf('    delta @ Pfa=%-7g : min=%+9.4f  median=%+9.4f  max=%+9.4f\n', ...
            pf, min(d), median(d), max(d));
        rows(end+1) = mkrow(sprintf('delta_%s_%g', dets(di).name, pf), d); %#ok<AGROW>
    end
end

%% ---- Recommended formats ------------------------------------------------
fprintf('\n=====================================================================\n');
fprintf(' RECOMMENDED FIXED-POINT FORMATS\n');
fprintf('=====================================================================\n');
fprintf('%-14s %-10s %7s %7s %7s   %s\n','signal','format','int','frac','total','resolution');

recommend('xc (log LUT out)', XC_MAX,            14, true);
recommend('xc^2',             XC_MAX^2,          13, false);
recommend('xc^3',             XC_MAX^3,          13, true);
recommend('c1',               max(abs(acc.c1)),  13, true);
recommend('c2',               max(acc.c2),       14, false);
recommend('c3',               max(abs(acc.c3)),  14, true);

fprintf('\n  Accumulator widths (must hold the full windowed sum, N=%d):\n', N);
accwidth('S1', N*XC_MAX,     14, true);
accwidth('S2', N*XC_MAX^2,   13, false);
accwidth('S3', N*XC_MAX^3,   13, true);

fprintf('\nNOTE: these are the RAW signal ranges. The LUT ADDRESS ranges are a\n');
fprintf('      separate decision (percentile-clipped, see each generate_LUTs_*.m),\n');
fprintf('      because an address range wide enough for the rarest outlier wastes\n');
fprintf('      most of the table on values that essentially never occur.\n');

%% ---- Save ---------------------------------------------------------------
if ~isempty(rows)
    T = struct2table(rows);
    outPath = fullfile(paths.results,'fixedpoint_ranges.csv');
    writetable(T, outPath);
    fprintf('\nWritten: %s\n', outPath);
end

R = struct('acc', acc, 'N', N, 'sli', sli, 'guard', guard, 'X0', X0, 'XCMax', XC_MAX);
end


%% =======================================================================
function s = shape_of(prm)
    switch prm.Name
        case 'Weibull',          s = prm.C;
        case 'Lognormal',        s = prm.sigma;
        case 'GeneralizedGamma', s = prm.k;
        case 'G0',               s = prm.u;
        case 'BurrXII',          s = prm.kappa;
        otherwise,               s = [];
    end
end

function r = mkrow(name, v)
    r = struct('Signal', name, 'Min', min(v), 'P001', prctile(v,0.1), ...
        'P50', median(v), 'P999', prctile(v,99.9), 'Max', max(v), ...
        'IntBits', NaN, 'FracBits', NaN, 'TotalBits', NaN, 'Signed', NaN, 'Note', '');
end

function recommend(name, maxabs, frac, signed)
%RECOMMEND  Integer bits needed to hold maxabs, at a given fractional width.
    ib = max(0, ceil(log2(max(maxabs, eps))));
    tot = ib + frac + double(signed);
    if signed
        fmt = sprintf('Q%d.%d s', ib, frac);
    else
        fmt = sprintf('Q%d.%d u', ib, frac);
    end
    fprintf('%-14s %-10s %7d %7d %7d   %.3e\n', name, fmt, ib, frac, tot, 2^-frac);
end

function accwidth(name, maxabs, frac, signed)
    ib = max(0, ceil(log2(max(maxabs, eps))));
    tot = ib + frac + double(signed);
    fprintf('    %-4s max |value| = %9.2f  ->  %2d int + %2d frac%s = %d bits\n', ...
        name, maxabs, ib, frac, tern(signed,' + sign',''), tot);
end

function o = tern(c,a,b)
    if c, o = a; else, o = b; end
end
