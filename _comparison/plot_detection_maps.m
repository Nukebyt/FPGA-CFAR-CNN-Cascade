function plot_detection_maps(varargin)
%PLOT_DETECTION_MAPS  Qualitative side-by-side: every detector on the same
%   scene, at the same window geometry, at a MATCHED MEASURED false-alarm rate.
%
%   PLOT_DETECTION_MAPS()
%   PLOT_DETECTION_MAPS('Image','000101.jpg', 'Sli',21, 'TargetPfa',1e-3)
%
%   ---------------------------------------------------------------------
%   WHY THE NOMINAL Pfa IS NOT USED HERE
%   ---------------------------------------------------------------------
%   Showing all five detectors at the same NOMINAL Pfa would be a picture of
%   their calibration error, not of their discrimination: on SSDD the measured
%   rates at a common nominal differ by more than an order of magnitude, so
%   one panel would be speckled and another blank, and the reader would
%   conclude the blank one is "better". Instead each detector's nominal Pfa is
%   searched (by bisection on log10 Pfa) until its MEASURED background
%   false-alarm rate hits the same target. Every panel then spends the same
%   false-alarm budget, and the only thing that differs between them is where
%   that budget gets spent -- which is the actual question.
%
%   Invalid windows are drawn in blue rather than left as background, because
%   a pixel where the estimator had no solution can never fire; conflating it
%   with correctly-rejected clutter is what makes a detector that failed to
%   run look like a well-calibrated one.
%
%   NAME-VALUE OPTIONS
%     'Image'     : SSDD filename (default: an image with several ships)
%     'Sli'       : window size (default 21)
%     'Guard'     : guard size (default 15)
%     'TargetPfa' : measured background false-alarm rate to match (default 1e-3)

p = inputParser;
addParameter(p, 'Image',     '');
addParameter(p, 'Sli',       21);
addParameter(p, 'Guard',     15);
addParameter(p, 'TargetPfa', 1e-3);
parse(p, varargin{:});

paths = cfar_setup();
S     = cfar_style();
sli   = p.Results.Sli;
guard = p.Results.Guard;
target = p.Results.TargetPfa;

%% ---- Pick a scene ------------------------------------------------------
imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
if isempty(p.Results.Image)
    % Prefer a multi-ship scene: a single-ship image cannot show the
    % difference between a detector that finds all targets and one that
    % finds one and a lot of clutter.
    name = '';
    for k = round(linspace(1, numel(imgs), 60))
        [~, b, ~] = fileparts(imgs(k).name);
        xp = fullfile(paths.ssddXml, [b '.xml']);
        if isfile(xp) && size(parseVOCBoxes(xp),1) >= 3
            name = imgs(k).name; break;
        end
    end
    if isempty(name), name = imgs(1).name; end
else
    name = p.Results.Image;
end

Iraw = imread(fullfile(paths.ssddJpeg, name));
if ndims(Iraw) == 3, Iraw = rgb2gray(Iraw); end
I = double(Iraw);
[~, base, ~] = fileparts(name);
gt = parseVOCBoxes(fullfile(paths.ssddXml, [base '.xml']));

fprintf('Scene: %s  (%dx%d, %d ships), sli=%d guard=%d, matching measured Pfa = %g\n', ...
    name, size(I,1), size(I,2), size(gt,1), sli, guard, target);

fe = cfar_front_end(I, sli, guard);
dets = detector_registry('Set','core');

%% ---- Match each detector to the same MEASURED Pfa ----------------------
maps = cell(numel(dets),1);
info = cell(numel(dets),1);

for di = 1:numel(dets)
    prm = dets(di).params(fe.c2, fe.c3, fe.skew);
    [dmap, usedPfa, gotPfa] = match_pfa(fe, prm, dets(di).tlog, gt, target);
    m = cfar_metrics(dmap, gt, prm.valid);
    maps{di} = struct('map', dmap, 'valid', prm.valid);
    info{di} = sprintf('%s\nnominal P_{fa}=%.1e -> measured %.1e\nP_d %.2f  |  FP objects %d  |  %.0f%% invalid', ...
        dets(di).name, usedPfa, gotPfa, m.Pd_ship, m.FP, 100*(1-mean(prm.valid(:))));
    fprintf('  %-12s nominal %.2e -> measured %.2e | Pd %.3f | TP %d FP %d FN %d | invalid %.1f%%\n', ...
        dets(di).name, usedPfa, gotPfa, m.Pd_ship, m.TP, m.FP, m.FN, ...
        100*(1-mean(prm.valid(:))));
end

%% ---- Figure ------------------------------------------------------------
f = figure('Color','w','Units','pixels','Position',[40 40 1500 460],'Visible','off');
nP = numel(dets) + 1;

ax = subplot(1,nP,1);
imshow(I, []); hold on;
for k = 1:size(gt,1)
    rectangle('Position',[gt(k,1) gt(k,2) gt(k,3)-gt(k,1) gt(k,4)-gt(k,2)], ...
        'EdgeColor',[0.93 0.63 0],'LineWidth',1.4);
end
hold off;
title(ax, sprintf('%s\n%d ground-truth ships (amber)', name, size(gt,1)), ...
    'FontWeight','normal','FontSize',10,'FontName',S.font);

for di = 1:numel(dets)
    ax = subplot(1,nP,di+1);
    rgb = repmat(mat2gray(I), [1 1 3]);
    R = rgb(:,:,1); G = rgb(:,:,2); B = rgb(:,:,3);
    dm = maps{di}.map;  iv = ~maps{di}.valid;
    R(iv) = 0.16; G(iv) = 0.47; B(iv) = 0.84;     % no MoLC solution
    R(dm) = 0.89; G(dm) = 0.29; B(dm) = 0.28;     % detection
    imshow(cat(3,R,G,B)); hold on;
    for k = 1:size(gt,1)
        rectangle('Position',[gt(k,1) gt(k,2) gt(k,3)-gt(k,1) gt(k,4)-gt(k,2)], ...
            'EdgeColor',[0.93 0.63 0],'LineWidth',1.0);
    end
    hold off;
    title(ax, info{di}, 'FontWeight','normal','FontSize',9,'FontName',S.font);
end

annotation(f,'textbox',[0.01 0.94 0.98 0.055],'String', ...
    sprintf(['Fig 8  Same scene, same window (sli=%d/guard=%d), same MEASURED false-alarm budget (P_{fa} = %g).' ...
             '  Red = detection, blue = estimator had no solution.'], sli, guard, target), ...
    'EdgeColor','none','FontName',S.font,'FontSize',12,'FontWeight','bold', ...
    'Color',S.ink,'VerticalAlignment','middle');
for a = findobj(f,'Type','axes')'
    pos = a.Position; a.Position = [pos(1) pos(2)*0.95 pos(3) pos(4)*0.86];
end

exportgraphics(f, fullfile(paths.figures,'08_detection_maps.png'), 'Resolution', 200);
exportgraphics(f, fullfile(paths.figures,'08_detection_maps.pdf'), 'ContentType','vector');
close(f);
fprintf('Figure written: 08_detection_maps.png / .pdf\n');
end


%% =======================================================================
function [dmap, usedPfa, gotPfa] = match_pfa(fe, prm, tlogFcn, gt, target)
%MATCH_PFA  Bisect on log10(nominal Pfa) until the MEASURED background
%   false-alarm rate reaches `target`.
%
%   The measured rate is monotone non-increasing in the nominal one (a smaller
%   requested Pfa can only raise the threshold), so bisection is well posed.
%   The bracket is deliberately very wide -- the whole point of this figure is
%   that the nominal values needed to reach a common measured rate differ by
%   orders of magnitude between detectors.
    lo = -12; hi = -0.05;      % log10 nominal Pfa bracket
    dmap = []; usedPfa = NaN; gotPfa = NaN;

    for it = 1:28
        mid = 0.5*(lo+hi);
        pf  = 10^mid;
        d   = decide(fe, prm, tlogFcn, pf);
        r   = measured_pfa(d, prm.valid, gt);

        dmap = d; usedPfa = pf; gotPfa = r;

        if ~isfinite(r) || r > target
            hi = mid;          % too many false alarms -> demand a smaller Pfa
        else
            lo = mid;
        end
    end
end

function d = decide(fe, prm, tlogFcn, Pfa)
    delta = tlogFcn(prm, Pfa);
    T = fe.c1 + delta;
    d = (fe.x > T) & prm.valid & isfinite(T);
end

function r = measured_pfa(dmap, valid, gt)
    [h,w] = size(dmap);
    ship = false(h,w);
    for k = 1:size(gt,1)
        c0 = max(1,round(gt(k,1))); r0 = max(1,round(gt(k,2)));
        c1 = min(w,round(gt(k,3))); r1 = min(h,round(gt(k,4)));
        ship(r0:r1, c0:c1) = true;
    end
    bg = ~ship & valid;
    if ~any(bg(:)), r = NaN; return; end
    r = sum(dmap(:) & bg(:)) / sum(bg(:));
end
