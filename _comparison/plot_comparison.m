function plot_comparison(varargin)
%PLOT_COMPARISON  Figures for the Phase 2 cross-detector comparison.
%
%   PLOT_COMPARISON()                 uses Results/comparison_raw_main.csv
%   PLOT_COMPARISON('Tag','_main')
%
%   Writes PNG + PDF to _comparison/Figures/.
%
%   ---------------------------------------------------------------------
%   THE FIGURE THAT MATTERS MOST IS FIGURE 2, NOT FIGURE 1
%   ---------------------------------------------------------------------
%   Figure 1 (Pd vs window size at a fixed NOMINAL Pfa) is the conventional
%   chart and it is actively misleading on this problem. The five detectors do
%   NOT hit the same operating point when handed the same nominal Pfa: on
%   SSDD, Weibull's measured false-alarm rate overshoots its nominal by well
%   over an order of magnitude while Lognormal's tracks it closely. Reading
%   "Weibull has the best Pd" off Figure 1 therefore compares a detector
%   running loose against one running tight, and says nothing about which
%   clutter model fits.
%
%   Figure 2 is the ROC view: Pd against the MEASURED false-alarm rate, with
%   nominal Pfa swept across six decades so every detector traces out its own
%   operating curve. That is the only fair comparison, and it is what the
%   detector ranking in FINDINGS.md is based on.
%
%   Figure 3 makes the calibration gap itself the subject: measured Pfa
%   against nominal. A detector on the diagonal has a clutter model that
%   actually describes the data.
%
%   Figures 4-6 cover the other half of "efficiency": estimator cost per
%   megapixel, the invalid-window fraction (a detector that cannot solve its
%   own estimator on half the windows is not achieving a low Pfa, it is
%   failing to run), and shape-parameter spread, which sets how finely the
%   Phase 3 ROM has to resolve the shape axis.

p = inputParser;
addParameter(p, 'Tag', '_main');
parse(p, varargin{:});
tag = p.Results.Tag;

paths = cfar_setup();
S = cfar_style();

rawPath = fullfile(paths.results, sprintf('comparison_raw%s.csv', tag));
if ~isfile(rawPath)
    error('plot_comparison:NoResults', 'Run run_comparison first -- %s not found.', rawPath);
end
T = readtable(rawPath);
Sm = readtable(fullfile(paths.results, sprintf('comparison_summary%s.csv', tag)));

if iscell(Sm.Detector), detNames = Sm.Detector; else, detNames = cellstr(Sm.Detector); end
Sm.Detector = detNames;

sliList = unique(Sm.Sli)';
pfaList = unique(Sm.Pfa)';
fprintf('Plotting from %s: %d rows, sli %s, Pfa %s\n', ...
    rawPath, height(T), mat2str(sliList), mat2str(pfaList));

%% =====================================================================
%% FIG 1 -- Pd vs window size, at fixed nominal Pfa  (the misleading one)
%% =====================================================================
f = newfig([980 420]);
pfaShow = [1e-2 1e-4];
for sp = 1:2
    ax = subplot(1,2,sp); hold(ax,'on');
    pf = pfaShow(sp);
    for d = S.core
        n = d{1};
        m = strcmp(Sm.Detector, n) & pfaeq(Sm.Pfa, pf);
        if ~any(m), continue; end
        [ss, o] = sort(Sm.Sli(m)); yy = Sm.Pd_pooled(m); yy = yy(o);
        plot(ax, ss, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
            'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor', 'w', ...
            'DisplayName', n);
    end
    xline(ax, 18, ':', 'Cyclone V ceiling', 'Color', [0.45 0.45 0.43], ...
        'LabelVerticalAlignment','bottom', 'LabelHorizontalAlignment','right', ...
        'FontSize', 9, 'HandleVisibility','off');
    dress(ax, S, 'Sliding window size (sli, px)', 'Pooled ship-level P_d', ...
        sprintf('Nominal P_{fa} = %g', pf));
    ylim(ax, [0 1]);
    if sp == 1, legend(ax, 'Location','southeast', 'Box','off', 'FontSize', 9, 'TextColor','k'); end
end
sgtitle_(f, 'Fig 1  Detection rate vs window size, at matched NOMINAL P_{fa}', ...
    'Conventional view -- but the detectors are NOT at the same operating point here; see Fig 3.', S);
savefig_(f, paths.figures, '01_pd_vs_window_nominal');

%% =====================================================================
%% FIG 2 -- ROC: Pd vs MEASURED Pfa  (the fair comparison)
%% =====================================================================
f = newfig([980 420]);
sliShow = [17 51];
for sp = 1:2
    ax = subplot(1,2,sp); hold(ax,'on');
    sl = sliShow(sp);
    for d = S.core
        n = d{1};
        m = strcmp(Sm.Detector, n) & Sm.Sli == sl & Sm.Pfa_pooled > 0;
        if ~any(m), continue; end
        [xx, o] = sort(Sm.Pfa_pooled(m)); yy = Sm.Pd_pooled(m); yy = yy(o);
        plot(ax, xx, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
            'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', ...
            'DisplayName', n);
    end
    set(ax, 'XScale', 'log');
    dress(ax, S, 'Measured pixel P_{fa} (pooled)', 'Pooled ship-level P_d', ...
        sprintf('sli = %d, guard = %d', sl, guard_for(Sm, sl)));
    ylim(ax, [0 1]);
    if sp == 1, legend(ax, 'Location','southeast', 'Box','off', 'FontSize', 9, 'TextColor','k'); end
end
sgtitle_(f, 'Fig 2  ROC -- detection rate vs ACTUAL false-alarm rate', ...
    'Nominal P_{fa} swept over six decades so every detector traces its own operating curve. This is the fair comparison.', S);
savefig_(f, paths.figures, '02_roc_measured');

%% =====================================================================
%% FIG 3 -- Pfa calibration: measured vs nominal
%% =====================================================================
f = newfig([980 420]);
for sp = 1:2
    ax = subplot(1,2,sp); hold(ax,'on');
    sl = sliShow(sp);
    lims = [min(pfaList)/3, 1];
    plot(ax, lims, lims, '-', 'Color', [0.6 0.6 0.58], 'LineWidth', 1.2, ...
        'DisplayName', 'perfect calibration');
    for d = S.core
        n = d{1};
        m = strcmp(Sm.Detector, n) & Sm.Sli == sl;
        if ~any(m), continue; end
        [xx, o] = sort(Sm.Pfa(m)); yy = Sm.Pfa_pooled(m); yy = yy(o);
        plot(ax, xx, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
            'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', ...
            'DisplayName', n);
    end
    set(ax, 'XScale','log', 'YScale','log');
    xlim(ax, lims); ylim(ax, lims);
    dress(ax, S, 'Nominal P_{fa} (requested)', 'Measured pixel P_{fa}', ...
        sprintf('sli = %d', sl));
    if sp == 1, legend(ax, 'Location','southeast', 'Box','off', 'FontSize', 9, 'TextColor','k'); end
end
sgtitle_(f, 'Fig 3  Does the clutter model actually fit? Measured vs requested P_{fa}', ...
    'On the grey diagonal = the model describes the clutter tail. Above it = the detector is running looser than it claims.', S);
savefig_(f, paths.figures, '03_pfa_calibration');

%% =====================================================================
%% FIG 4 -- Computational cost
%% =====================================================================
f = newfig([980 420]);

ax = subplot(1,2,1); hold(ax,'on');
for d = S.core
    n = d{1};
    m = strcmp(Sm.Detector, n) & pfaeq(Sm.Pfa, 1e-4);
    if ~any(m), continue; end
    [ss,o] = sort(Sm.Sli(m)); yy = Sm.EstMsPerMP(m); yy = yy(o);
    plot(ax, ss, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
        'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', 'DisplayName', n);
end
set(ax,'YScale','log');
dress(ax, S, 'Sliding window size (sli, px)', 'Estimator time (ms per megapixel)', ...
    'Parameter estimation only -- shared front end excluded');
legend(ax, 'Location','northeast', 'Box','off', 'FontSize', 9, 'TextColor','k');

ax = subplot(1,2,2); hold(ax,'on');
for d = S.core
    n = d{1};
    m = strcmp(Sm.Detector, n) & pfaeq(Sm.Pfa, 1e-4);
    if ~any(m), continue; end
    [ss,o] = sort(Sm.Sli(m)); yy = 100*Sm.FractionInvalid(m); yy = yy(o);
    plot(ax, ss, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
        'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', 'DisplayName', n);
end
dress(ax, S, 'Sliding window size (sli, px)', 'Windows with no MoLC solution (%)', ...
    'Estimator failure rate');
ylim(ax,[0 100]);
sgtitle_(f, 'Fig 4  Computational efficiency and estimator robustness', ...
    'Right panel is not a footnote: a detector that cannot solve its estimator cannot declare a target, so it posts a low P_{fa} by not running.', S);
savefig_(f, paths.figures, '04_cost_and_validity');

%% =====================================================================
%% FIG 5 -- F1 (the ranking metric) vs window size
%% =====================================================================
f = newfig([980 420]);
for sp = 1:2
    ax = subplot(1,2,sp); hold(ax,'on');
    pf = pfaShow(sp);
    for d = S.core
        n = d{1};
        m = strcmp(Sm.Detector, n) & pfaeq(Sm.Pfa, pf);
        if ~any(m), continue; end
        [ss,o] = sort(Sm.Sli(m)); yy = Sm.F1_pooled(m); yy = yy(o);
        plot(ax, ss, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
            'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', 'DisplayName', n);
    end
    xline(ax, 18, ':', 'Color', [0.45 0.45 0.43], 'HandleVisibility','off');
    dress(ax, S, 'Sliding window size (sli, px)', 'Object-level F1 (pooled)', ...
        sprintf('Nominal P_{fa} = %g', pf));
    if sp == 1, legend(ax, 'Location','northwest', 'Box','off', 'FontSize', 9, 'TextColor','k'); end
end
sgtitle_(f, 'Fig 5  Object-level F1 -- the ranking metric', ...
    'Detections clustered into objects; a cluster overlapping a ground-truth box is a hit. Penalises both misses and clutter breakthrough.', S);
savefig_(f, paths.figures, '05_f1_vs_window');

%% =====================================================================
%% FIG 6 -- Variants: what the cheap/exact alternatives cost
%% =====================================================================
f = newfig([980 420]);

ax = subplot(1,2,1); hold(ax,'on');
pairs = {'GenGamma','GenGamma-exact'; 'G0','G0-L1'};
for i = 1:numel(pairs)
    n = pairs{i};
    m = strcmp(Sm.Detector, n) & pfaeq(Sm.Pfa, 1e-4);
    if ~any(m), continue; end
    [ss,o] = sort(Sm.Sli(m)); yy = Sm.Pd_pooled(m); yy = yy(o);
    plot(ax, ss, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
        'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', 'DisplayName', n);
end
dress(ax, S, 'Sliding window size (sli, px)', 'Pooled ship-level P_d', 'P_d, nominal P_{fa} = 10^{-4}');
ylim(ax,[0 1]);
legend(ax, 'Location','best', 'Box','off', 'FontSize', 9, 'TextColor','k');

ax = subplot(1,2,2); hold(ax,'on');
for i = 1:numel(pairs)
    n = pairs{i};
    m = strcmp(Sm.Detector, n) & pfaeq(Sm.Pfa, 1e-4);
    if ~any(m), continue; end
    [ss,o] = sort(Sm.Sli(m)); yy = 100*Sm.FractionInvalid(m); yy = yy(o);
    plot(ax, ss, yy, S.lines(n), 'Color', S.colors(n), 'LineWidth', S.lw, ...
        'Marker', S.markers(n), 'MarkerSize', S.ms, 'MarkerFaceColor','w', 'DisplayName', n);
end
dress(ax, S, 'Sliding window size (sli, px)', 'Windows with no MoLC solution (%)', 'Estimator failure rate');
ylim(ax,[0 100]);
sgtitle_(f, 'Fig 6  Solver variants -- the exact/cheap alternatives', ...
    'GenGamma-exact inverts the true MoLC relation instead of its cubic approximation; G0-L1 fixes L=1 for a 1-D ROM.', S);
savefig_(f, paths.figures, '06_variants');

fprintf('Figures written to %s\n', paths.figures);
end


%% =======================================================================
function f = newfig(sz)
    f = figure('Color','w', 'Units','pixels', 'Position',[80 80 sz(1) sz(2)], ...
               'Visible','off');
end

function dress(ax, S, xl, yl, ttl)
%DRESS  Recessive grid and axes, thin marks in front.
    ax.Color = 'w';   % axes plot-area background; NOT inherited from the
                       % figure's 'Color' -- left unset it silently follows
                       % this install's app/OS theme (dark, here), so it must
                       % be forced explicitly on every axes.
    grid(ax,'on');
    ax.GridColor = S.grid;  ax.GridAlpha = 1;  ax.Layer = 'bottom';
    ax.MinorGridColor = S.grid; ax.MinorGridAlpha = 0.6;
    ax.XColor = S.axcol; ax.YColor = S.axcol;
    ax.FontName = S.font; ax.FontSize = S.fontsz - 1;
    ax.Box = 'off'; ax.TickDir = 'out'; ax.LineWidth = 0.8;
    xlabel(ax, xl, 'FontSize', S.fontsz, 'Color', S.ink);
    ylabel(ax, yl, 'FontSize', S.fontsz, 'Color', S.ink);
    title(ax, ttl, 'FontSize', S.fontsz, 'FontWeight','normal', 'Color', S.ink);
end

function sgtitle_(f, ttl, sub, S)
%SGTITLE_  Headline plus a one-line takeaway, so each figure states its own
%   conclusion rather than leaving it to the reader.
    annotation(f, 'textbox', [0.02 0.935 0.96 0.055], 'String', ttl, ...
        'EdgeColor','none', 'FontName', S.font, 'FontSize', S.fontsz+2, ...
        'FontWeight','bold', 'Color', S.ink, 'VerticalAlignment','middle');
    annotation(f, 'textbox', [0.02 0.895 0.96 0.045], 'String', sub, ...
        'EdgeColor','none', 'FontName', S.font, 'FontSize', S.fontsz-1, ...
        'Color', [0.32 0.32 0.30], 'VerticalAlignment','middle');
    % leave room for the two-line header
    axs = findobj(f,'Type','axes');
    for a = axs'
        pos = a.Position;
        a.Position = [pos(1) pos(2)*1.02 pos(3) pos(4)*0.80];
    end
end

function savefig_(f, dir_, name)
    exportgraphics(f, fullfile(dir_, [name '.png']), 'Resolution', 200);
    exportgraphics(f, fullfile(dir_, [name '.pdf']), 'ContentType','vector');
    close(f);
    fprintf('  wrote %s.png / .pdf\n', name);
end

function m = pfaeq(col, target)
%PFAEQ  Match a nominal-Pfa column to a target with a relative tolerance.
%   The values round-trip through CSV, so exact floating-point equality is
%   not safe to rely on.
    m = abs(col - target) <= 1e-9 * target;
end

function g = guard_for(Sm, sl)
    m = Sm.Sli == sl;
    g = Sm.Guard(find(m,1));
    if isempty(g), g = NaN; end
end
