function results = run_comparison(varargin)
%RUN_COMPARISON  Phase 2: cross-detector efficiency sweep over sliding-window
%   sizes and false-alarm rates, on the SSDD dataset.
%
%   results = RUN_COMPARISON()
%   results = RUN_COMPARISON('NumImages', 300, 'Sli', [21 51], ...)
%
%   Sweeps a 4-way grid of (image, window geometry, detector, Pfa) and writes
%   a per-cell results table plus pooled summaries to _comparison/Results/.
%
%   ---------------------------------------------------------------------
%   HOW THE WORK IS FACTORED  (this is what makes the grid tractable)
%   ---------------------------------------------------------------------
%   The three stages of a CFAR detector depend on different parts of the grid:
%       cfar_front_end   (image, sli, guard)            -- shared by ALL
%                                                          detectors and ALL Pfa
%       estimator        (image, sli, guard, detector)  -- shared by all Pfa
%       threshold+decide (everything)
%   so the loops are nested in exactly that order and each stage runs the
%   minimum number of times. Calling a monolithic detector function inside the
%   innermost loop would recompute the front end 28x and each estimator 4x per
%   image per geometry. The same factoring is what the FPGA has, so the timing
%   breakdown reported here maps straight onto the hardware block diagram.
%
%   ---------------------------------------------------------------------
%   WINDOW GEOMETRIES
%   ---------------------------------------------------------------------
%   The (sli, guard) pairs are not invented here. They are the pairs the
%   existing Weibull project already established as best-guard-for-that-sli on
%   this dataset (README section 2.7), extended down to the small windows the
%   Cyclone V can actually fit. Guard sits at 0.65-0.80 of sli because SSDD
%   ship boxes have median max-dimension ~37 px: a small guard lets a ship's
%   own bright pixels leak into its reference window and corrupt the very
%   statistics meant to describe the clutter around it.
%
%   sli = 17 is included specifically to bracket the hardware's SLI = 18,
%   which is the largest window that fits Cyclone V under default
%   optimization and which cfar_front_end cannot take directly because it
%   requires an odd window. sli = 51 is the software-optimal point. The gap
%   between those two is the accuracy price of the current device.
%
%   ---------------------------------------------------------------------
%   WHAT "EFFICIENCY" MEANS HERE -- both senses are measured
%   ---------------------------------------------------------------------
%   Detection efficiency : ship-level Pd, measured-vs-nominal Pfa, and
%       object-level precision / recall / F1 (see cfar_metrics). F1 is the
%       ranking metric; Pd alone rewards a detector that floods the image.
%   Computational efficiency : front-end, estimator and threshold time
%       reported separately, plus the estimator's INVALID-WINDOW FRACTION.
%       That last one is not a footnote: a detector with no MoLC solution on
%       half its windows cannot declare a target there, so it posts an
%       excellent false-alarm rate by not running. Pfa and coverage have to
%       be read together, which is why cfar_metrics excludes invalid pixels
%       from the background denominator and reports CoverageFraction.
%
%   NAME-VALUE OPTIONS
%     'NumImages' : how many SSDD images to sample, evenly spaced through the
%                   1160 (default 150). The full set is used for the final
%                   re-run of the winning configuration, not for the grid.
%     'Sli'       : window sizes (default [11 15 17 21 31 41 51])
%     'Guard'     : matching guards; default is derived from Sli by the table
%                   above and must be the same length if given explicitly.
%     'Pfa'       : default [1e-3 1e-4 1e-5 1e-6]
%     'Detectors' : 'all' (default) | 'core'
%     'Tag'       : filename suffix for the output CSVs (default '')
%     'Random'    : true | false (default false) -- sample NumImages uniformly
%                   at random (via randperm, then sorted for readable logs)
%                   instead of evenly-spaced. Evenly-spaced remains the
%                   default so every prior sweep/finding in FINDINGS.md stays
%                   reproducible; use Random for an unbiased-sample-style
%                   report where "randomly selected images" is the point.
%     'Seed'      : RNG seed for Random (default 42) -- fixed so a random
%                   sweep is still exactly reproducible.
%
%   OUTPUT
%     results : table, one row per (image, geometry, detector, Pfa) cell.
%     Also written: Results/comparison_raw<Tag>.csv and
%                   Results/comparison_summary<Tag>.csv

p = inputParser;
addParameter(p, 'NumImages', 150);
addParameter(p, 'Sli',       [11 15 17 21 31 41 51]);
addParameter(p, 'Guard',     []);
addParameter(p, 'Pfa',       [1e-3 1e-4 1e-5 1e-6]);
addParameter(p, 'Detectors', 'all');
addParameter(p, 'Tag',       '');
addParameter(p, 'Random',    false);
addParameter(p, 'Seed',      42);
parse(p, varargin{:});

paths = cfar_setup();

sliList = p.Results.Sli(:)';
if isempty(p.Results.Guard)
    guardList = default_guard(sliList);
else
    guardList = p.Results.Guard(:)';
    if numel(guardList) ~= numel(sliList)
        error('run_comparison:GuardLength', 'Guard must match Sli in length.');
    end
end
PfaList = p.Results.Pfa(:)';
dets    = detector_registry('Set', p.Results.Detectors);
tag     = p.Results.Tag;

%% ---- Image selection ---------------------------------------------------
imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
if isempty(imgs)
    error('run_comparison:NoImages', 'No JPEGs found in %s', paths.ssddJpeg);
end
nImg = min(p.Results.NumImages, numel(imgs));
if p.Results.Random
    rng(p.Results.Seed);
    imgIdx = sort(randperm(numel(imgs), nImg));
else
    % Evenly spaced rather than the first N: SSDD is ordered, and the first few
    % hundred are not representative of the inshore/offshore mix.
    imgIdx = unique(round(linspace(1, numel(imgs), nImg)));
end

fprintf('=====================================================================\n');
fprintf(' Phase 2 comparison sweep\n');
fprintf('   images    : %d of %d (evenly spaced)\n', numel(imgIdx), numel(imgs));
fprintf('   geometries: %s\n', strjoin(arrayfun(@(a,b) sprintf('%d/%d',a,b), ...
    sliList, guardList, 'UniformOutput', false), ', '));
fprintf('   Pfa       : %s\n', mat2str(PfaList));
fprintf('   detectors : %s\n', strjoin({dets.name}, ', '));
fprintf('   grid cells: %d\n', numel(imgIdx)*numel(sliList)*numel(dets)*numel(PfaList));
fprintf('=====================================================================\n');

%% ---- Preallocate --------------------------------------------------------
nRows = numel(imgIdx) * numel(sliList) * numel(dets) * numel(PfaList);
R = struct( ...
    'Image', cell(nRows,1), 'Detector', cell(nRows,1), ...
    'Sli', cell(nRows,1), 'Guard', cell(nRows,1), 'N', cell(nRows,1), ...
    'Pfa', cell(nRows,1), 'NMoments', cell(nRows,1), ...
    'Pd_ship', cell(nRows,1), 'DetectedShips', cell(nRows,1), 'TotalShips', cell(nRows,1), ...
    'Pfa_actual', cell(nRows,1), 'FalsePixels', cell(nRows,1), 'BackgroundPixels', cell(nRows,1), ...
    'TP', cell(nRows,1), 'FP', cell(nRows,1), 'FN', cell(nRows,1), ...
    'Precision', cell(nRows,1), 'Recall', cell(nRows,1), 'F1', cell(nRows,1), ...
    'NumClusters', cell(nRows,1), 'NumDetections', cell(nRows,1), ...
    'FractionInvalid', cell(nRows,1), 'CoverageFraction', cell(nRows,1), ...
    'LogDomainMismatchPct', cell(nRows,1), ...
    'MedianShape', cell(nRows,1), 'IQRShape', cell(nRows,1), ...
    'TimeFrontEndMs', cell(nRows,1), 'TimeEstimatorMs', cell(nRows,1), ...
    'TimeThresholdMs', cell(nRows,1), 'Megapixels', cell(nRows,1), ...
    'MedianC2', cell(nRows,1), 'MedianSkew', cell(nRows,1));
row = 0;

tSweep = tic;

%% ---- The grid ----------------------------------------------------------
for ii = 1:numel(imgIdx)
    imgName = imgs(imgIdx(ii)).name;
    [~, base, ~] = fileparts(imgName);

    Iraw = imread(fullfile(paths.ssddJpeg, imgName));
    if ndims(Iraw) == 3, Iraw = rgb2gray(Iraw); end
    I = double(Iraw);
    [h, w] = size(I);
    MP = h*w/1e6;

    xmlPath = fullfile(paths.ssddXml, [base '.xml']);
    if ~isfile(xmlPath)
        warning('run_comparison:NoXML', 'No annotation for %s -- skipped.', base);
        continue;
    end
    gt = parseVOCBoxes(xmlPath);

    for gi = 1:numel(sliList)
        sli = sliList(gi); guard = guardList(gi);

        tfe = tic;
        fe = cfar_front_end(I, sli, guard);
        tFrontEnd = toc(tfe) * 1000;

        medC2   = median(fe.c2(:));
        medSkew = median(fe.skew(:));

        for di = 1:numel(dets)
            % CA-CFAR-style baseline entries (params==[]) don't factor into
            % a Pfa-independent estimator step -- see detector_registry.m's
            % header. Fall back to calling .fcn directly per Pfa for those;
            % every other detector keeps the factored front-end/estimator/
            % threshold split unchanged.
            monolithic = isempty(dets(di).params);

            if ~monolithic
                tes = tic;
                prm = dets(di).params(fe.c2, fe.c3, fe.skew);
                tEst = toc(tes) * 1000;
                [medShape, iqrShape] = shape_summary(prm);
            else
                tEst = 0;
                medShape = NaN; iqrShape = NaN;
            end

            for pi = 1:numel(PfaList)
                Pfa = PfaList(pi);

                if ~monolithic
                    tth = tic;
                    delta = dets(di).tlog(prm, Pfa);
                    T_log = fe.c1 + delta;
                    valid = prm.valid & isfinite(T_log);
                    det_map = (fe.x > T_log) & valid;
                    tThr = toc(tth) * 1000;

                    % amplitude-domain cross-check, same as cfar_compose
                    det_amp = (I > exp(2*T_log) - 0.5) & valid;
                    mismPct = 100 * sum(det_amp(:) ~= det_map(:)) / numel(det_map);
                else
                    tth = tic;
                    [det_map, ~, prm, ~, ld] = dets(di).fcn(I, sli, guard, Pfa, 'FrontEnd', fe);
                    tThr = toc(tth) * 1000;
                    valid = prm.valid;
                    mismPct = ld.PercentMismatch;
                end

                m = cfar_metrics(det_map, gt, valid);

                row = row + 1;
                R(row).Image = imgName;              R(row).Detector = dets(di).name;
                R(row).Sli = sli;                    R(row).Guard = guard;
                R(row).N = fe.N;                     R(row).Pfa = Pfa;
                R(row).NMoments = dets(di).nmom;
                R(row).Pd_ship = m.Pd_ship;          R(row).DetectedShips = m.DetectedShips;
                R(row).TotalShips = m.TotalShips;    R(row).Pfa_actual = m.Pfa_actual;
                R(row).FalsePixels = m.FalsePixels;  R(row).BackgroundPixels = m.BackgroundPixels;
                R(row).TP = m.TP;                    R(row).FP = m.FP;
                R(row).FN = m.FN;                    R(row).Precision = m.Precision;
                R(row).Recall = m.Recall;            R(row).F1 = m.F1;
                R(row).NumClusters = m.NumClusters;  R(row).NumDetections = m.NumDetections;
                R(row).FractionInvalid = 1 - mean(prm.valid(:));
                R(row).CoverageFraction = m.CoverageFraction;
                R(row).LogDomainMismatchPct = mismPct;
                R(row).MedianShape = medShape;       R(row).IQRShape = iqrShape;
                R(row).TimeFrontEndMs = tFrontEnd;   R(row).TimeEstimatorMs = tEst;
                R(row).TimeThresholdMs = tThr;       R(row).Megapixels = MP;
                R(row).MedianC2 = medC2;             R(row).MedianSkew = medSkew;
            end
        end
    end

    if mod(ii, 10) == 0 || ii == numel(imgIdx)
        el = toc(tSweep);
        fprintf('  [%3d/%3d] %s   elapsed %6.1f s, est. total %6.1f s\n', ...
            ii, numel(imgIdx), imgName, el, el * numel(imgIdx) / ii);
    end
end

R = R(1:row);
results = struct2table(R);

%% ---- Write raw + summary ----------------------------------------------
rawPath = fullfile(paths.results, sprintf('comparison_raw%s.csv', tag));
writetable(results, rawPath);

summary = summarise_sweep(results);
sumPath = fullfile(paths.results, sprintf('comparison_summary%s.csv', tag));
writetable(summary, sumPath);

fprintf('\n=====================================================================\n');
fprintf(' Sweep complete in %.1f s (%d rows)\n', toc(tSweep), height(results));
fprintf('   raw     : %s\n', rawPath);
fprintf('   summary : %s\n', sumPath);
fprintf('=====================================================================\n');

print_comparison_headline(summary, PfaList);
end
