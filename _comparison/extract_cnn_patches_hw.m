function extract_cnn_patches_hw(varargin)
%EXTRACT_CNN_PATCHES_HW  Full-prevalence, hardware-geometry candidate dataset
%   for Paper 2's CNN discriminator.
%
%   Supersedes extract_cnn_patches.m for anything that reports cascade
%   numbers.  Two differences from that script, both found 2026-10-02:
%
%   (1) GEOMETRY.  extract_cnn_patches.m runs the Weibull prescreen at
%       sli=61/guard=49 (software peak-Pd), but the DE10 RTL is SLI=17/
%       GUARD=13 (the window that fits Cyclone V next to a CNN).  The CNN
%       must be trained on the candidate stream the DEPLOYED prescreen
%       produces.  Defaults here are the hardware's.
%
%   (2) PREVALENCE.  extract_cnn_patches.m caps negatives at NegPosRatio=8x
%       positives per image.  Measured on 12 images, Weibull at Pfa=1e-3
%       really emits ~1,200 (sli=61) / ~1,730 (sli=17) false-alarm clusters
%       per image against ~24 ship-overlapping clusters: 1.3-2.0% positive,
%       not the 12.6% of the capped dataset.  Every precision / candidate-
%       reduction figure computed on the capped set overstates the cascade.
%       This script keeps EVERY cluster.
%
%   Patches are stored as uint8 (log-amplitude fe.x quantized over
%   [XLo, XHi]) -- the CNN's input is 8-bit in hardware anyway, so training
%   on these is hardware-faithful, and 2M patches fit in ~3 GB.
%   Clusters near the image edge are kept (symmetric padding) rather than
%   dropped.
%
%   OUTPUT  Results/cnn_patches_hw.mat (-v7.3)
%     patches  : P x P x N uint8      (dequantize: XLo + q*(XHi-XLo)/255)
%     labels   : N x 1 logical        (cluster overlaps >=1 GT box)
%     imgIdx   : N x 1 uint16         (index into imgNames)
%     gtIdx    : N x 1 uint8          (first GT box hit, 0 for negatives)
%     area, bboxH, bboxW : N x 1      (cluster pixel area / bbox, in px)
%     peakX, meanX : N x 1 single     (max / mean of fe.x inside cluster)
%     c1, c2   : N x 1 single         (prescreen local log-mean / variance
%                                      at the centroid -- free in hardware)
%     cxy      : N x 2 single         (centroid, x then y)
%     imgNames : cellstr; imgNShips : GT box count per image (recall
%                denominators include ships CFAR never touched)
%     meta     : struct of the settings used

p = inputParser;
addParameter(p, 'NumImages', 1200);
addParameter(p, 'Sli', 17);
addParameter(p, 'Guard', 13);
addParameter(p, 'Pfa', 1e-3);
addParameter(p, 'PatchSize', 40);
addParameter(p, 'Seed', 42);
addParameter(p, 'XLo', -0.40);
addParameter(p, 'XHi', 2.80);
addParameter(p, 'OutName', 'cnn_patches_hw.mat');
addParameter(p, 'Pool', 1);        % integer dxd average (round-half-up) applied to the stored patch -- the deployed net's down-sampled input
addParameter(p, 'GateTau', -Inf);   % keep a NEGATIVE only if (mean of centre 3x3 of fe.x) - c1 >= GateTau; positives always kept
parse(p, varargin{:});

paths = cfar_setup();
sli = p.Results.Sli; guard = p.Results.Guard; Pfa = p.Results.Pfa;
P = p.Results.PatchSize; halfP = P/2;
XLo = p.Results.XLo; XHi = p.Results.XHi;

imgs = dir(fullfile(paths.hrsidImages, '*.png'));
nImg = min(p.Results.NumImages, numel(imgs));
rng(p.Results.Seed);
imgSel = sort(randperm(numel(imgs), nImg));   % same draw as extract_cnn_patches.m
boxMap = load_coco_boxes(paths.hrsidAnnFile);

fprintf('extract_cnn_patches_hw: sli=%d guard=%d Pfa=%g, %d images, patch %d\n', ...
    sli, guard, Pfa, nImg, P);

cP = cell(nImg,1); cL = cell(nImg,1); cG = cell(nImg,1);
cF = cell(nImg,1);     % [area bboxH bboxW peakX meanX c1 c2 cx cy gate3x3]
imgNames = cell(nImg,1); imgNShips = zeros(nImg,1);
tStart = tic;
for ii = 1:nImg
    name = imgs(imgSel(ii)).name;
    imgNames{ii} = name;
    if ~isKey(boxMap, name), continue; end
    gt = boxMap(name);                    % [xmin ymin xmax ymax]
    imgNShips(ii) = size(gt,1);

    I = double(imread(fullfile(paths.hrsidImages, name)));
    if ndims(I) == 3, I = I(:,:,1); end
    [h, w] = size(I);

    fe = cfar_front_end(I, sli, guard);
    prm = WeibullCFAR_Params(fe.c2);
    delta = WeibullCFAR_TLog(prm, Pfa);
    det = (fe.x > fe.c1 + delta) & prm.valid;

    cc = bwconncomp(det, 8);
    n = cc.NumObjects;
    if n == 0, continue; end
    L = labelmatrix(cc);
    rp = regionprops(cc, fe.x, 'Area', 'Centroid', 'BoundingBox', 'MaxIntensity', 'MeanIntensity');

    % identical overlap test to cfar_metrics.m / extract_cnn_patches.m:
    % a cluster is a ship if ANY of its pixels lies inside a GT box
    gtHit = zeros(n,1,'uint8');
    for k = 1:size(gt,1)
        c0 = max(1, round(gt(k,1))); r0 = max(1, round(gt(k,2)));
        c1b = min(w, round(gt(k,3))); r1 = min(h, round(gt(k,4)));
        ids = unique(L(r0:r1, c0:c1b));
        ids = ids(ids > 0);
        upd = ids(gtHit(ids) == 0);
        gtHit(upd) = k;
    end

    xp = padarray(single(fe.x), [halfP halfP], 'symmetric');
    c1p = single(fe.c1); c2p = single(fe.c2);
    cen = reshape([rp.Centroid], 2, n)';          % [x y]
    bb  = reshape([rp.BoundingBox], 4, n)';       % [x y w h]
    Pst = P / p.Results.Pool;
    pat = zeros(Pst, Pst, n, 'uint8');
    cf  = zeros(n, 9, 'single');
    keep = true(n,1);
    gateAll = zeros(n,1,'single');
    gateTau = p.Results.GateTau;
    for ci = 1:n
        cy = min(max(round(cen(ci,2)), 1), h); cx = min(max(round(cen(ci,1)), 1), w);
        g3 = mean(xp(cy+halfP-1:cy+halfP+1, cx+halfP-1:cx+halfP+1), 'all') - c1p(cy,cx);
        gateAll(ci) = g3;
        if gtHit(ci) == 0 && isfinite(gateTau)
            if g3 < gateTau, keep(ci) = false; continue; end
        end
        crop = xp(cy+1 : cy+P, cx+1 : cx+P);       % padded coords: orig + halfP
        q8 = round(min(max((crop - XLo) / (XHi - XLo), 0), 1) * 255);
        if p.Results.Pool > 1
            dd = p.Results.Pool;
            q8 = squeeze(sum(sum(reshape(q8, dd, P/dd, dd, P/dd), 1), 3));
            q8 = floor((q8 + dd*dd/2) / (dd*dd));
        end
        pat(1:size(q8,1),1:size(q8,2),ci) = uint8(q8);
        cf(ci,:) = [rp(ci).Area, bb(ci,4), bb(ci,3), rp(ci).MaxIntensity, ...
                    rp(ci).MeanIntensity, c1p(cy,cx), c2p(cy,cx), cen(ci,1), cen(ci,2)];
    end
    cP{ii} = pat(:,:,keep); cL{ii} = gtHit(keep) > 0; cG{ii} = gtHit(keep); cF{ii} = [cf(keep,:), gateAll(keep)];

    if mod(ii, 50) == 0
        fprintf('  [%4d/%4d] elapsed %.0fs  clusters so far %d\n', ii, nImg, toc(tStart), ...
            sum(cellfun(@(c) size(c,1), cL)));
    end
end

% ---- assemble (per-image cell -> flat arrays) ----
nPer = cellfun(@(c) numel(c), cL);
N = sum(nPer);
fprintf('Total %d clusters (%d ship-overlapping, %.2f%%) over %d images\n', ...
    N, sum(cellfun(@sum, cL)), 100*sum(cellfun(@sum, cL))/N, nImg);

patches = cat(3, cP{:}); cP = [];
labels  = vertcat(cL{:});
gtIdx   = vertcat(cG{:});
F       = vertcat(cF{:});
imgIdx  = uint16(repelem((1:nImg)', nPer));
area  = uint16(F(:,1)); bboxH = uint16(F(:,2)); bboxW = uint16(F(:,3));
peakX = F(:,4); meanX = F(:,5); c1 = F(:,6); c2 = F(:,7); cxy = F(:,8:9); gate3 = F(:,10);
meta = struct('Sli', sli, 'Guard', guard, 'Pfa', Pfa, 'PatchSize', P, ...
    'NumImages', nImg, 'DatasetSource', 'HRSID', 'XLo', XLo, 'XHi', XHi, ...
    'Prevalence', 'full (no negative cap); negatives below GateTau dropped', 'Seed', p.Results.Seed, ...
    'GateTau', p.Results.GateTau, 'Pool', p.Results.Pool);

outPath = fullfile(paths.results, p.Results.OutName);
save(outPath, 'patches', 'labels', 'imgIdx', 'gtIdx', 'area', 'bboxH', 'bboxW', ...
    'peakX', 'meanX', 'c1', 'c2', 'cxy', 'gate3', 'imgNames', 'imgNShips', 'meta', '-v7.3');
fprintf('Wrote %s  (%.0fs)\n', outPath, toc(tStart));
end
