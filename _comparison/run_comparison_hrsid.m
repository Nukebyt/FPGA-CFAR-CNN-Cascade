function results = run_comparison_hrsid(varargin)
%RUN_COMPARISON_HRSID  Phase 2 cross-detector sweep, on HRSID instead of SSDD.
%
%   results = RUN_COMPARISON_HRSID()
%   results = RUN_COMPARISON_HRSID('NumImages', 300, 'Sli', [21 51], ...)
%
%   Same 4-way grid (image, window geometry, detector, Pfa) and the same
%   pooling/metrics as run_comparison.m -- see that file's header for the
%   full rationale. This one exists because HRSID's images are lossless PNG
%   (verified: 8-bit, R==G==B on every pixel, i.e. true grayscale SAR
%   amplitude with no JPEG compression), while SSDD's are JPEG. Running the
%   identical sweep on both is how a JPEG-compression artifact gets told
%   apart from a real windowing effect -- see the SLI-plateau confound this
%   was built to resolve (ROADMAP.md Phase 6, RESEARCH-CHECKLIST Sec 5).
%
%   HRSID also has real internal diversity SSDD doesn't: 3 sensors
%   (Sentinel-1B, TerraSAR-X, TanDEM-X), 3 polarizations (HH/VV/HH... /HV),
%   incident angles 18-60 degrees, inshore+offshore scenes, 0.5-3m
%   resolution. Ground truth is COCO-format polygon+bbox (see
%   load_coco_boxes.m); only the bbox is used here, same as SSDD's VOC boxes.
%
%   NAME-VALUE OPTIONS  (identical meaning to run_comparison.m)
%     'NumImages' : how many HRSID images to sample, evenly spaced through
%                   the 5604 (default 150).
%     'Sli'       : window sizes (default [11 15 17 21 31 41 51])
%     'Guard'     : matching guards; default from default_guard.m
%     'Pfa'       : default [1e-3 1e-4 1e-5 1e-6]
%     'Detectors' : 'all' (default) | 'core'
%     'Tag'       : filename suffix for the output CSVs (default '_hrsid')
%     'Random'    : true | false (default false) -- sample NumImages uniformly
%                   at random (via randperm, then sorted for readable logs)
%                   instead of evenly-spaced. See run_comparison.m's header
%                   for the same option -- kept identical across both so a
%                   paired SSDD/HRSID sweep is drawn the same way.
%     'Seed'      : RNG seed for Random (default 42).
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
addParameter(p, 'Tag',       '_hrsid');
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
        error('run_comparison_hrsid:GuardLength', 'Guard must match Sli in length.');
    end
end
PfaList = p.Results.Pfa(:)';
dets    = detector_registry('Set', p.Results.Detectors);
tag     = p.Results.Tag;

%% ---- Image selection + ground truth -------------------------------------
imgs = dir(fullfile(paths.hrsidImages, '*.png'));
if isempty(imgs)
    error('run_comparison_hrsid:NoImages', 'No PNGs found in %s', paths.hrsidImages);
end
nImg = min(p.Results.NumImages, numel(imgs));
if p.Results.Random
    rng(p.Results.Seed);
    imgIdx = sort(randperm(numel(imgs), nImg));
else
    % Evenly spaced, same reasoning as SSDD: HRSID's file order groups by
    % source panorama (Table 2 of the HRSID paper), so the first N are not
    % representative of the sensor/polarization mix.
    imgIdx = unique(round(linspace(1, numel(imgs), nImg)));
end

fprintf('Loading HRSID annotations (train_test2017.json, 5604 images, 16951 ships)...\n');
boxMap = load_coco_boxes(paths.hrsidAnnFile);

fprintf('=====================================================================\n');
fprintf(' Phase 2 comparison sweep -- HRSID\n');
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

    Iraw = imread(fullfile(paths.hrsidImages, imgName));
    % Verified R==G==B on every pixel (true grayscale in an RGB PNG
    % container) -- take one channel rather than rgb2gray's luma weighting,
    % which would be a no-op here but costs a needless assumption.
    if ndims(Iraw) == 3, Iraw = Iraw(:,:,1); end
    I = double(Iraw);
    [h, w] = size(I);
    MP = h*w/1e6;

    if ~isKey(boxMap, imgName)
        warning('run_comparison_hrsid:NoAnnotation', ...
            'No annotation entry for %s -- skipped.', imgName);
        continue;
    end
    gt = boxMap(imgName);

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
            % header and run_comparison.m's identical branch.
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
