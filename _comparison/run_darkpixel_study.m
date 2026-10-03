function Tout = run_darkpixel_study(varargin)
%RUN_DARKPIXEL_STUDY  Supplementary Phase 2 experiment: how much of the
%   3-parameter detectors' estimator failure is caused by 8-bit quantization
%   at the dark end of the image, rather than by clutter texture?
%
%   Tout = RUN_DARKPIXEL_STUDY()
%
%   ---------------------------------------------------------------------
%   THE QUESTION
%   ---------------------------------------------------------------------
%   Both 3-parameter detectors depend on the third log-cumulant c3 only
%   through the dimensionless log-skewness s = c3/c2^1.5 (Generalized Gamma
%   through r = s^2, Burr XII through s itself), and both have a support
%   condition on it:
%       Burr XII        : -1.139443 < s < 2
%       Gen. Gamma      : s^2 < 8 (cubic solver) or < 4 (exact solver)
%   The main sweep finds those conditions failing on a large fraction of SSDD
%   windows. The question is whether that is a real property of the clutter or
%   an artefact of the imagery's 8-bit encoding.
%
%   ---------------------------------------------------------------------
%   IT IS AN ARTEFACT -- compare_datasets.m already established that
%   ---------------------------------------------------------------------
%   Running the same front end over both datasets (Results/dataset_comparison.csv):
%
%       dataset                       median log-skew   % outside Burr support
%       MSTAR, native complex SAR          -0.413                0.4%
%       MSTAR, rounded to 8 bits           -0.725                4.5%
%       SSDD, 8-bit JPEG                   -1.420               56.1%
%
%   Same X-band SAR amplitude in every row; only the encoding differs. So the
%   support failures are a property of SSDD's encoding, not of SAR clutter.
%   This script answers the follow-up question: WHICH part of the encoding,
%   and can a change to the log LUT recover the lost performance?
%
%   ---------------------------------------------------------------------
%   THE MECHANISM
%   ---------------------------------------------------------------------
%   x = log(sqrt(I+0.5)) is very steep at the bottom of the 8-bit range:
%       I = 0 -> x = -0.347,  I = 1 -> +0.203,  I = 2 -> +0.458
%   against a typical SSDD clutter level of x ~ 2. One I=0 pixel in a 216-cell
%   window is therefore a ~2.3-unit outlier, and c3 weights outliers by the
%   CUBE. A handful of near-black pixels can dominate the third moment while
%   barely moving the second.
%
%   ---------------------------------------------------------------------
%   THE EXPERIMENT
%   ---------------------------------------------------------------------
%   Stratify windows by their local near-black pixel fraction and compare the
%   skewness distribution and support-failure rate across strata; then re-run
%   the two 3-parameter detectors with cfar_front_end's 'IFloor' option, which
%   clamps I up to a floor before the log and so removes the cliff. IFloor is
%   FREE in hardware -- it only rewrites the bottom few entries of the
%   256-entry log-amplitude ROM, changing no logic at all -- so if it recovers
%   a meaningful amount of Pd/F1 it should simply be adopted in Phase 3.
%
%   NAME-VALUE OPTIONS
%     'NumImages' (60), 'Sli' (21), 'Guard' (15), 'Pfa' (1e-2)
%     'Floors'    : IFloor values to test (default [0 1 2 4])
%
%   OUTPUT: table written to Results/darkpixel_study.csv

p = inputParser;
addParameter(p, 'NumImages', 60);
addParameter(p, 'Sli',       21);
addParameter(p, 'Guard',     15);
addParameter(p, 'Pfa',       1e-2);
addParameter(p, 'Floors',    [0 1 2 4]);
parse(p, varargin{:});

paths = cfar_setup();
S = cfar_style();

sli   = p.Results.Sli;
guard = p.Results.Guard;
Pfa   = p.Results.Pfa;
floors = p.Results.Floors;

imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
idx  = unique(round(linspace(1, numel(imgs), min(p.Results.NumImages, numel(imgs)))));

dets = detector_registry();
keep = ismember({dets.name}, {'Weibull','Lognormal','GenGamma','G0','BurrXII'});
dets = dets(keep);

BURR_LO = -1.139443;

fprintf('=====================================================================\n');
fprintf(' Dark-pixel study: %d images, sli=%d guard=%d, Pfa=%g, IFloor in %s\n', ...
    numel(idx), sli, guard, Pfa, mat2str(floors));
fprintf('=====================================================================\n');

%% ---- Part A: skewness stratified by local near-black fraction -----------
fprintf('\n--- A. Log-skewness vs local near-black pixel fraction (IFloor = 0) ---\n');

SK = []; DK = []; C2 = [];
tk = (sli-1)/2;
for k = idx
    Ik = imread(fullfile(paths.ssddJpeg, imgs(k).name));
    if ndims(Ik) == 3, Ik = rgb2gray(Ik); end
    Id = double(Ik);
    fe = cfar_front_end(Id, sli, guard);
    dark = conv2(padarray(double(Id <= 2), [tk tk], 'symmetric'), ...
                 ones(sli)/sli^2, 'valid');
    SK = [SK; fe.skew(1:9:end)']; %#ok<AGROW>
    DK = [DK; dark(1:9:end)'];    %#ok<AGROW>
    C2 = [C2; fe.c2(1:9:end)'];   %#ok<AGROW>
end
SK = SK(:); DK = DK(:); C2 = C2(:);

edges = [0 0.005 0.02 0.05 0.15 0.40 1.001];
fprintf('%22s %10s %11s %10s %14s %14s\n', 'near-black fraction', 'windows', ...
    'med skew', 'med c2', '% below Burr', '% r>4 (GG ex)');
strata = struct('lo',{},'hi',{},'n',{},'medSkew',{},'medC2',{},'pBurr',{},'pGG',{});
for b = 1:numel(edges)-1
    m = DK >= edges(b) & DK < edges(b+1);
    if sum(m) < 100, continue; end
    st = struct('lo', edges(b), 'hi', edges(b+1), 'n', sum(m), ...
        'medSkew', median(SK(m)), 'medC2', median(C2(m)), ...
        'pBurr', 100*mean(SK(m) < BURR_LO), 'pGG', 100*mean(SK(m).^2 > 4));
    strata(end+1) = st; %#ok<AGROW>
    fprintf('  [%5.3f, %5.3f)      %10d %11.3f %10.4f %13.1f%% %13.1f%%\n', ...
        st.lo, st.hi, st.n, st.medSkew, st.medC2, st.pBurr, st.pGG);
end
fprintf('\n  Pure Weibull clutter has s = %.6f EXACTLY -- that is Burr''s lower limit,\n', BURR_LO);
fprintf('  so any stratum with a median well below it is more negatively log-skewed\n');
fprintf('  than Weibull clutter can be.\n');

%% ---- Part B: detector performance vs IFloor -----------------------------
fprintf('\n--- B. Detector performance vs IFloor ---\n');
fprintf('%-12s %8s %9s %11s %10s %10s\n', 'Detector','IFloor','Pd','Pfa_meas','F1','invalid%');

rows = struct('Detector',{},'IFloor',{},'Pd',{},'PfaMeasured',{},'F1',{}, ...
              'FractionInvalid',{},'ShipsDetected',{},'ShipsTotal',{});

for fl = floors
    % accumulate pooled counters across images for each detector
    acc = zeros(numel(dets), 7);   % [shipsDet shipsTot falsePx bgPx TP FP FN]
    invAcc = zeros(numel(dets), 1);
    nImgUsed = 0;

    for k = idx
        Ik = imread(fullfile(paths.ssddJpeg, imgs(k).name));
        if ndims(Ik) == 3, Ik = rgb2gray(Ik); end
        Id = double(Ik);
        [~, base, ~] = fileparts(imgs(k).name);
        xmlPath = fullfile(paths.ssddXml, [base '.xml']);
        if ~isfile(xmlPath), continue; end
        gt = parseVOCBoxes(xmlPath);

        fe = cfar_front_end(Id, sli, guard, 'IFloor', fl);
        nImgUsed = nImgUsed + 1;

        for di = 1:numel(dets)
            prm   = dets(di).params(fe.c2, fe.c3, fe.skew);
            delta = dets(di).tlog(prm, Pfa);
            T_log = fe.c1 + delta;
            valid = prm.valid & isfinite(T_log);
            dmap  = (fe.x > T_log) & valid;

            m = cfar_metrics(dmap, gt, valid);
            acc(di,:) = acc(di,:) + [m.DetectedShips m.TotalShips ...
                m.FalsePixels m.BackgroundPixels m.TP m.FP m.FN];
            invAcc(di) = invAcc(di) + (1 - mean(prm.valid(:)));
        end
    end

    for di = 1:numel(dets)
        a = acc(di,:);
        Pd  = a(1)/max(a(2),1);
        pf  = a(3)/max(a(4),1);
        pr  = a(5)/max(a(5)+a(6),1);
        rc  = a(5)/max(a(5)+a(7),1);
        F1  = 2*pr*rc/max(pr+rc, eps);
        inv = invAcc(di)/max(nImgUsed,1);
        fprintf('%-12s %8d %9.3f %11.2e %10.3f %9.1f%%\n', ...
            dets(di).name, fl, Pd, pf, F1, 100*inv);
        rows(end+1) = struct('Detector', dets(di).name, 'IFloor', fl, ...
            'Pd', Pd, 'PfaMeasured', pf, 'F1', F1, 'FractionInvalid', inv, ...
            'ShipsDetected', a(1), 'ShipsTotal', a(2)); %#ok<AGROW>
    end
    fprintf('\n');
end

Tout = struct2table(rows);
outPath = fullfile(paths.results, 'darkpixel_study.csv');
writetable(Tout, outPath);
fprintf('Written: %s\n', outPath);

%% ---- Figure -------------------------------------------------------------
f = figure('Color','w','Units','pixels','Position',[80 80 980 400],'Visible','off');

ax = subplot(1,2,1); hold(ax,'on');
bx = [strata.lo];
plot(ax, bx, [strata.medSkew], '-o', 'Color', [42 120 214]/255, 'LineWidth', 1.8, ...
    'MarkerSize', 6, 'MarkerFaceColor','w', 'DisplayName', 'median log-skewness');
yline(ax, BURR_LO, '--', 'Burr XII support limit (= pure Weibull clutter)', ...
    'Color', [235 104 52]/255, 'LineWidth', 1.4, 'FontSize', 9, ...
    'LabelHorizontalAlignment','left', 'HandleVisibility','off');
set(ax,'XScale','log');
dressax(ax, S, 'Local near-black (I<=2) pixel fraction', 'Median log-skewness s', ...
    'A. Where the extreme skew comes from');
legend(ax,'Location','southeast','Box','off','FontSize',9,'TextColor','k');

ax = subplot(1,2,2); hold(ax,'on');
for d = {'GenGamma','BurrXII','Weibull','Lognormal'}
    n = d{1};
    m = strcmp(Tout.Detector, n);
    if ~any(m), continue; end
    plot(ax, Tout.IFloor(m), Tout.F1(m), S.lines(n), 'Color', S.colors(n), ...
        'LineWidth', 1.8, 'Marker', S.markers(n), 'MarkerSize', 6, ...
        'MarkerFaceColor','w', 'DisplayName', n);
end
dressax(ax, S, 'IFloor (intensity clamped up to this before the log)', ...
    'Object-level F1 (pooled)', sprintf('B. Effect of the fix (sli=%d, P_{fa}=%g)', sli, Pfa));
legend(ax,'Location','best','Box','off','FontSize',9,'TextColor','k');

annotation(f,'textbox',[0.02 0.93 0.96 0.06],'String', ...
    'Supplementary  Is the 3-parameter estimators'' failure rate a clutter property or an 8-bit artefact?', ...
    'EdgeColor','none','FontName',S.font,'FontSize',13,'FontWeight','bold','Color',S.ink);
for a = findobj(f,'Type','axes')'
    pos = a.Position; a.Position = [pos(1) pos(2) pos(3) pos(4)*0.85];
end
exportgraphics(f, fullfile(paths.figures, '07_darkpixel_study.png'), 'Resolution', 200);
exportgraphics(f, fullfile(paths.figures, '07_darkpixel_study.pdf'), 'ContentType','vector');
close(f);
fprintf('Figure written: 07_darkpixel_study.png / .pdf\n');
end

function dressax(ax, S, xl, yl, ttl)
    ax.Color = 'w';   % force the plot-area background; not inherited from
                      % the figure's 'Color' on this install's dark theme
    grid(ax,'on'); ax.GridColor = S.grid; ax.GridAlpha = 1; ax.Layer = 'bottom';
    ax.XColor = S.axcol; ax.YColor = S.axcol; ax.Box = 'off'; ax.TickDir = 'out';
    ax.FontName = S.font; ax.FontSize = S.fontsz-1; ax.LineWidth = 0.8;
    xlabel(ax, xl, 'FontSize', S.fontsz, 'Color', S.ink);
    ylabel(ax, yl, 'FontSize', S.fontsz, 'Color', S.ink);
    title(ax, ttl, 'FontSize', S.fontsz, 'FontWeight','normal', 'Color', S.ink);
end
