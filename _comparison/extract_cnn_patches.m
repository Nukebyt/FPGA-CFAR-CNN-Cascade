function extract_cnn_patches(varargin)
%EXTRACT_CNN_PATCHES  Build the labeled patch dataset for Paper 2's CNN
%   discriminator stage, from real Weibull-CFAR detections on HRSID.
%
%   extract_cnn_patches('NumImages', 1000, ...)
%
%   Runs the same Weibull prescreen + 8-connected clustering this project's
%   cfar_metrics.m already uses for its object-level TP/FP/FN accounting,
%   but instead of collapsing each cluster to a pass/fail count, crops a
%   fixed-size patch around every cluster's centroid from the SAME
%   log-amplitude image cfar_front_end.m produces (fe.x -- the exact
%   representation the Weibull threshold decision itself is made on, so the
%   CNN sees the same domain the prescreen already committed to), and labels
%   it:
%
%       label = 1 (real ship)   if the cluster overlaps a ground-truth box
%               0 (false alarm) otherwise
%
%   using the identical per-cluster overlap test cfar_metrics.m's object-
%   level loop already implements (not a re-derivation).
%
%   Deliberately run at a LOOSE Pfa (default 1e-3, not the 1e-6 this
%   project's calibration work uses), at sli=61 -- Weibull's own measured
%   HRSID peak-Pd geometry (FINDINGS.md F12, Pd=0.979) -- because a cascade's
%   whole point is that the CFAR stage does not need to be stingy: it can
%   run loose (few missed ships) and pass everything through, false alarms
%   included, for the CNN stage to clean up. A tight Pfa would starve this
%   dataset of the false-alarm examples the discriminator most needs to see.
%
%   OUTPUT
%     Results/cnn_patches.mat -- struct with fields:
%       patches  : PatchSize x PatchSize x N single, log-amplitude, NOT yet
%                  normalized (mean/std reported separately so the exact
%                  same normalization can be reproduced in the RTL fixed-
%                  point pipeline later)
%       labels   : N x 1 logical (true = real ship)
%       imgName  : N x 1 cellstr, source image per patch (for a clean,
%                  image-disjoint train/val/test split -- never split by
%                  patch, since patches from the same image share clutter
%                  statistics and would leak)
%       clusterArea, centroid : per-patch cluster metadata, for diagnostics
%       meta     : Sli, Guard, Pfa, PatchSize, NumImages, DatasetSource

p = inputParser;
addParameter(p, 'NumImages', 1000);
addParameter(p, 'Sli', 61);
addParameter(p, 'Guard', []);
addParameter(p, 'Pfa', 1e-3);
addParameter(p, 'PatchSize', 32);
addParameter(p, 'Random', true);
addParameter(p, 'Seed', 42);
addParameter(p, 'NegPosRatio', 8);   % cap negatives per image at this multiple of that image's positives (at least 1 kept if any exist), to bound class imbalance and dataset size
parse(p, varargin{:});

paths = cfar_setup();
sli = p.Results.Sli;
if isempty(p.Results.Guard)
    guard = default_guard(sli);
else
    guard = p.Results.Guard;
end
Pfa = p.Results.Pfa;
P = p.Results.PatchSize;
halfP = P / 2;

imgs = dir(fullfile(paths.hrsidImages, '*.png'));
if isempty(imgs)
    error('extract_cnn_patches:NoImages', 'No PNGs found in %s', paths.hrsidImages);
end
nImg = min(p.Results.NumImages, numel(imgs));
if p.Results.Random
    rng(p.Results.Seed);
    imgIdx = sort(randperm(numel(imgs), nImg));
else
    imgIdx = unique(round(linspace(1, numel(imgs), nImg)));
end

fprintf('Loading HRSID annotations...\n');
boxMap = load_coco_boxes(paths.hrsidAnnFile);

fprintf('=====================================================================\n');
fprintf(' CNN patch extraction -- Weibull prescreen, sli=%d/guard=%d, Pfa=%g\n', sli, guard, Pfa);
fprintf(' Images: %d of %d, patch size %dx%d\n', nImg, numel(imgs), P, P);
fprintf('=====================================================================\n');

patches   = zeros(P, P, 0, 'single');
labels    = false(0, 1);
imgName   = {};
clusterArea = zeros(0, 1);
centroidXY  = zeros(0, 2);

tStart = tic;
nPos = 0; nNeg = 0;

for ii = 1:numel(imgIdx)
    name = imgs(imgIdx(ii)).name;
    if ~isKey(boxMap, name)
        continue;
    end
    gt = boxMap(name);  % [xmin ymin xmax ymax]

    Iraw = imread(fullfile(paths.hrsidImages, name));
    if ndims(Iraw) == 3, Iraw = Iraw(:,:,1); end
    I = double(Iraw);
    [h, w] = size(I);

    fe = cfar_front_end(I, sli, guard);
    prm = WeibullCFAR_Params(fe.c2);
    delta = WeibullCFAR_TLog(prm, Pfa);
    Tlog = fe.c1 + delta;
    detection_map = (fe.x > Tlog) & prm.valid;

    cc = bwconncomp(detection_map, 8);
    if cc.NumObjects == 0
        continue;
    end

    n_ships = size(gt, 1);
    boxRC = zeros(n_ships, 4);  % [r0 r1 c0 c1]
    for k = 1:n_ships
        c0 = max(1, round(gt(k,1))); r0 = max(1, round(gt(k,2)));
        c1 = min(w, round(gt(k,3))); r1 = min(h, round(gt(k,4)));
        boxRC(k,:) = [r0 r1 c0 c1];
    end

    % Two passes: first collect every cluster's label + geometry cheaply,
    % then decide which negatives to keep (capped) before paying for any
    % patch crop -- avoids wasting time cropping clusters that get dropped.
    hitBoxAll = false(cc.NumObjects, 1);
    cyAll = zeros(cc.NumObjects, 1); cxAll = zeros(cc.NumObjects, 1);
    for ci = 1:cc.NumObjects
        idx = cc.PixelIdxList{ci};
        [rr, ccol] = ind2sub([h w], idx);
        cyAll(ci) = mean(rr); cxAll(ci) = mean(ccol);
        if n_ships > 0
            for k = 1:n_ships
                b = boxRC(k,:);
                if any(rr >= b(1) & rr <= b(2) & ccol >= b(3) & ccol <= b(4))
                    hitBoxAll(ci) = true;
                    break;
                end
            end
        end
    end

    posIdx = find(hitBoxAll);
    negIdx = find(~hitBoxAll);
    nKeepNeg = max(1, round(p.Results.NegPosRatio * max(numel(posIdx), 1)));
    if numel(negIdx) > nKeepNeg
        negIdx = negIdx(randperm(numel(negIdx), nKeepNeg));
    end
    keepIdx = [posIdx; negIdx];

    for ci = keepIdx(:)'
        idx = cc.PixelIdxList{ci};
        cy = cyAll(ci); cx = cxAll(ci);
        hitBox = hitBoxAll(ci);

        r0 = round(cy - halfP) + 1; r1 = r0 + P - 1;
        c0 = round(cx - halfP) + 1; c1 = c0 + P - 1;
        if r0 < 1 || c0 < 1 || r1 > h || c1 > w
            continue;  % skip clusters too close to the image edge for a full patch
        end

        patch = single(fe.x(r0:r1, c0:c1));
        patches(:,:,end+1) = patch; %#ok<AGROW>
        labels(end+1,1) = hitBox; %#ok<AGROW>
        imgName{end+1,1} = name; %#ok<AGROW>
        clusterArea(end+1,1) = numel(idx); %#ok<AGROW>
        centroidXY(end+1,:) = [cx, cy]; %#ok<AGROW>

        if hitBox, nPos = nPos + 1; else, nNeg = nNeg + 1; end
    end

    if mod(ii, 100) == 0
        fprintf('  [%4d/%4d] %s  patches so far: %d pos / %d neg  elapsed %.1fs\n', ...
            ii, numel(imgIdx), name, nPos, nNeg, toc(tStart));
    end
end

fprintf('\nDone. %d positive (real ship), %d negative (false alarm) patches, %d total.\n', ...
    nPos, nNeg, nPos + nNeg);

meta = struct('Sli', sli, 'Guard', guard, 'Pfa', Pfa, 'PatchSize', P, ...
    'NumImages', nImg, 'DatasetSource', 'HRSID');

outPath = fullfile(paths.results, 'cnn_patches.mat');
save(outPath, 'patches', 'labels', 'imgName', 'clusterArea', 'centroidXY', 'meta', '-v7.3');
fprintf('Wrote %s\n', outPath);

end
