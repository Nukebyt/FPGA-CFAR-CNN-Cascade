function extract_cnn_patches_hwspec(varargin)
%EXTRACT_CNN_PATCHES_HWSPEC  Candidate dataset that follows the HARDWARE cascade spec exactly.
%
%   Spec the RTL implements (rtl/cascade/), reproduced here sample for sample:
%     1. q8   = QROM[pixel]  = floor(clip((log(sqrt(I+0.5)) - XLo)/(XHi-XLo), 0, 1)*255 + 0.5)
%     2. P    = 2x2 average of q8 on the frame grid, round-half-up: (a+b+c+d+2)>>2
%               (the "pooled frame store"; the CNN never sees full-resolution pixels)
%     3. D    = Weibull detection (float model here, sli/guard of the RTL, Pfa)
%     4. trigger T(y,x) = D(y,x) & ~D(y,x-1) & ~D(y-1,x-1) & ~D(y-1,x) & ~D(y-1,x+1)
%               = the left end of a detected run that has no detected neighbour above or to the left.
%               One streaming pass, needs only the previous row's detect bits.
%     5. gate g = x(trigger) - c1(trigger)  (both already exist at the trigger pixel in the Weibull
%               pipeline); an event is kept if g >= GateTau.
%     6. event position (j,i) = (floor(y/2), floor(x/2)) on the pooled grid (0-based)
%     7. patch = P(j-16 : j+15, i-16 : i+15), indices clamped to the frame (edge replicate)
%   Labels: a trigger is a ship if its 8-connected detection component overlaps a GT box (same
%   overlap test as cfar_metrics.m).  gtIdx = first GT box hit (0 = false alarm).
%
%   Positives are always kept; negatives only if g >= GateTau (so GateTau must be <= the deployed
%   gate).  Per-image trigger/component counts are saved for true candidate-rate accounting.
%
%   OUT: Results/<OutName> (-v7.3): patches 32x32xN uint8, labels, imgIdx, gtIdx, gate (g), tyx,
%        compArea, c1, xt, imgNames, imgNShips, imgNTrig, imgNComp, meta.

p = inputParser;
addParameter(p, 'NumImages', 5604);
addParameter(p, 'Sli', 17); addParameter(p, 'Guard', 13); addParameter(p, 'Pfa', 1e-3);
addParameter(p, 'XLo', -0.40); addParameter(p, 'XHi', 2.80);
addParameter(p, 'GateTau', -Inf);
addParameter(p, 'StorePatches', true);
addParameter(p, 'Fixed', false);      % true: bit-exact fixed-point detection/x/c1 (the RTL's arithmetic) instead of the float model
addParameter(p, 'ImgList', []);       % optional list of directory indices to process (default: first NumImages)
addParameter(p, 'OutName', 'cnn_patches_hwspec.mat');
parse(p, varargin{:});
paths = cfar_setup();
sli = p.Results.Sli; guard = p.Results.Guard; Pfa = p.Results.Pfa;
XLo = p.Results.XLo; XHi = p.Results.XHi; tau = p.Results.GateTau;
HALF = 16;                                   % patch = 32x32 on the pooled grid

imgs = dir(fullfile(paths.hrsidImages, '*.png'));
nImg = min(p.Results.NumImages, numel(imgs));
if ~isempty(p.Results.ImgList), imgOrder = p.Results.ImgList(:)'; else, imgOrder = 1:nImg; end
if p.Results.Fixed
    cfgFx = fixedpoint_config(sli, guard);
    sharedDir = fullfile(paths.root, 'lut', 'shared');
    wl = load(fullfile(paths.root, 'lut', 'weibull', 'weibull_delta_lut.mat'));
    pfaTable = [1e-3 1e-4 1e-5 1e-6]; pfaIdx = find(abs(pfaTable - Pfa) < 1e-12, 1);
    if isempty(pfaIdx), error('Fixed mode: Pfa must be one of the hardware planes %s', mat2str(pfaTable)); end
    TK = (sli-1)/2;
end
boxMap = load_coco_boxes(paths.hrsidAnnFile);
fprintf('hwspec extraction: %d images, sli=%d guard=%d Pfa=%g GateTau=%g\n', nImg, sli, guard, Pfa, tau);

cP = cell(numel(imgs),1); cM = cell(numel(imgs),1);
imgNames = cell(numel(imgs),1); imgNShips = zeros(numel(imgs),1); imgNTrig = zeros(numel(imgs),1); imgNComp = zeros(numel(imgs),1);
t0 = tic;
for ii = imgOrder
    name = imgs(ii).name; imgNames{ii} = name;
    if ~isKey(boxMap, name), continue; end
    gt = boxMap(name); imgNShips(ii) = size(gt,1);
    I = double(imread(fullfile(paths.hrsidImages, name)));
    if ndims(I) == 3, I = I(:,:,1); end
    h = floor(size(I,1)/2)*2; w = floor(size(I,2)/2)*2; I = I(1:h,1:w);

    if p.Results.Fixed
        fe = cfar_front_end_fixed(I, sli, guard, sharedDir, 'Config', cfgFx);
        nlut = wl.entries_per_pfa; c2r = fe.c2(:);
        aidx = min(max(round((c2r - wl.addr_min) / (wl.addr_max - wl.addr_min) * (nlut-1)), 0), nlut-1);
        dn = double(wl.code((pfaIdx-1)*nlut + aidx + 1));
        dc = sign(dn) .* floor((abs(dn) + 2) / 4);                       % rshift_round(.,2)
        c1c = double(fe.c1_code(:)); c1t = sign(c1c) .* floor((abs(c1c) + 4) / 8);   % rshift_round(.,3)
        Tc = (c1t + dc - 1242) * 16;
        D = reshape(double(fe.x_code(:)) > Tc, size(I));
        v = false(size(I)); v(TK+1:end-TK, TK+1:end-TK) = true;
        D = D & v; D(TK+1, TK+1) = false;                                % the RTL never emits the first interior pixel
        gq14 = double(fe.x_code) + 19866 - 2*double(fe.c1_code);          % hardware gate, Q.14
        fe.gfix = gq14 / 16384;
        fe.xf = fe.x; fe.c1f = fe.c1;
    else
        fe  = cfar_front_end(I, sli, guard);
        prm = WeibullCFAR_Params(fe.c2);
        D = (fe.x > fe.c1 + WeibullCFAR_TLog(prm, Pfa)) & prm.valid;
        fe.gfix = fe.x - fe.c1;
    end
    cc = bwconncomp(D, 8);
    imgNComp(ii) = cc.NumObjects;
    if cc.NumObjects == 0, continue; end
    L = labelmatrix(cc);
    area = cellfun(@numel, cc.PixelIdxList)';

    % ground-truth overlap per component
    gtHit = zeros(cc.NumObjects,1,'uint8');
    for k = 1:size(gt,1)
        c0 = max(1, round(gt(k,1))); r0 = max(1, round(gt(k,2)));
        c1b = min(w, round(gt(k,3))); r1 = min(h, round(gt(k,4)));
        ids = unique(L(r0:r1, c0:c1b)); ids = ids(ids > 0);
        upd = ids(gtHit(ids) == 0); gtHit(upd) = k;
    end

    % trigger map (streaming NMS on the detection plane)
    up   = [false(1,w); D(1:end-1,:)];
    prvL = [false(h,1), D(:,1:end-1)];
    upL  = [false(h,1), up(:,1:end-1)];
    upR  = [up(:,2:end), false(h,1)];
    T = D & ~prvL & ~up & ~upL & ~upR;
    [ty, tx] = find(T);
    nT = numel(ty); imgNTrig(ii) = nT;
    if nT == 0, continue; end
    lin = sub2ind([h w], ty, tx);
    lab = L(lin);
    hit = gtHit(lab); lbl = hit > 0;
    g = single(fe.gfix(lin));
    keep = lbl | (g >= tau);
    if ~any(keep), continue; end

    if p.Results.StorePatches
        q = floor(min(max((0.5*log(I + 0.5) - XLo) / (XHi - XLo), 0), 1) * 255 + 0.5);   % QROM
        P = floor((q(1:2:end,1:2:end) + q(1:2:end,2:2:end) + q(2:2:end,1:2:end) + q(2:2:end,2:2:end) + 2) / 4);
        hp = h/2; wp = w/2;
        ki = find(keep);
        pat = zeros(2*HALF, 2*HALF, numel(ki), 'uint8');
        for n = 1:numel(ki)
            j = floor((ty(ki(n))-1)/2); i = floor((tx(ki(n))-1)/2);          % 0-based pooled position
            rr = min(max((j-HALF):(j+HALF-1), 0), hp-1) + 1;
            cc2 = min(max((i-HALF):(i+HALF-1), 0), wp-1) + 1;
            pat(:,:,n) = uint8(P(rr, cc2));
        end
        cP{ii} = pat;
    end
    ki = find(keep);
    cM{ii} = [double(lbl(ki)), double(hit(ki)), double(g(ki)), double(ty(ki)), double(tx(ki)), ...
              double(area(lab(ki))), double(fe.c1(lin(ki))), double(fe.x(lin(ki))), double(imgIdxOf(ii, numel(ki)))];
    if mod(find(imgOrder==ii,1), 200) == 0
        fprintf('  [%4d/%4d] %.0fs  events so far %d\n', ii, nImg, toc(t0), sum(cellfun(@(c) size(c,1), cM)));
    end
end

M = vertcat(cM{:});
labels = logical(M(:,1)); gtIdx = uint8(M(:,2)); gate = single(M(:,3)); tyx = single(M(:,4:5));
compArea = uint16(M(:,6)); c1 = single(M(:,7)); xt = single(M(:,8)); imgIdx = uint16(M(:,9));
if p.Results.StorePatches, patches = cat(3, cP{:}); else, patches = zeros(32,32,0,'uint8'); end
fprintf('Total %d events (%d ship-overlapping = %.2f%%)\n', numel(labels), sum(labels), 100*mean(labels));
meta = struct('Sli', sli, 'Guard', guard, 'Pfa', Pfa, 'XLo', XLo, 'XHi', XHi, 'GateTau', tau, 'Fixed', p.Results.Fixed, ...
    'Spec', 'hwspec: NMS trigger, pooled frame store, edge-replicate 32x32 window, gate=x-c1', 'NumImages', nImg);
save(fullfile(paths.results, p.Results.OutName), 'patches', 'labels', 'imgIdx', 'gtIdx', 'gate', 'tyx', ...
    'compArea', 'c1', 'xt', 'imgNames', 'imgNShips', 'imgNTrig', 'imgNComp', 'meta', '-v7.3');
fprintf('Wrote %s (%.0fs)\n', p.Results.OutName, toc(t0));
end

function v = imgIdxOf(ii, n)
v = repmat(ii, n, 1);
end
