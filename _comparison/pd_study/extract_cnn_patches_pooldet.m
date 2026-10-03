function extract_cnn_patches_pooldet(varargin)
%EXTRACT_CNN_PATCHES_POOLDET  Candidate dataset for the cascade whose prescreen runs on the POOLED image (Paper 2, phase 5 prototype).
%
%   Same outputs / field names as extract_cnn_patches_hwspec.m, so cnn/hwlib.py can train on it (variant "pooldet*"), but the candidate events come from
%   the new prescreen of pd_variants_eval2.m:  pool factor f (2 or 4), pooling mode (int: mean of intensity | log: mean of log amplitude), window (sli, guard),
%   Weibull Pfa, contrast gate x - c1 >= Gate, event type (nms | peak3 | peak5 | comp).  Border padded.
%   Event position (pooled-domain pixel) -> full-resolution centre -> pooled-2x2 store coordinates (j, i); the CNN patch is the 32x32 window of the
%   pooled-2x2 store (mean of the QROM codes, exactly the hardware pooled store) around (j, i), edge replicated -- identical to the existing cascade.
%   Label: the event lies within TolPx full-resolution pixels of a ship polygon.  gtIdx = that ship's index (0 = false alarm).
%
%   usage: extract_cnn_patches_pooldet('Pool',2,'Mode','log','Sli',25,'Guard',17,'Pfa',0.03,'Gate',0.6,'Event','peak5','Range',[1 1900],'OutName','cnn_patches_pooldetA_p1.mat')

p = inputParser;
addParameter(p, 'Pool', 2); addParameter(p, 'Mode', 'log'); addParameter(p, 'Sli', 25); addParameter(p, 'Guard', 17);
addParameter(p, 'Pfa', 0.03); addParameter(p, 'Gate', 0.6); addParameter(p, 'Event', 'peak5'); addParameter(p, 'TolPx', 4);
addParameter(p, 'Range', [1 5604]); addParameter(p, 'OutName', 'cnn_patches_pooldet_p1.mat'); addParameter(p, 'StorePatches', true); addParameter(p, 'Ctx', false);
parse(p, varargin{:});
q = p.Results;
paths = cfar_setup();
XLo = -0.40; XHi = 2.80; HALF = 16; X0 = 1.2125;
f = q.Pool; TK = (q.Sli - 1) / 2; TG = (q.Guard - 1) / 2;
Gp = struct('sli', q.Sli, 'guard', q.Guard, 'pad', true);

raw = jsondecode(fileread(paths.hrsidAnnFile));
id2name = containers.Map('KeyType', 'double', 'ValueType', 'char');
for i = 1:numel(raw.images), id2name(raw.images(i).id) = raw.images(i).file_name; end
byName = containers.Map('KeyType', 'char', 'ValueType', 'any'); anns = raw.annotations;
for i = 1:numel(anns)
    nm = id2name(anns(i).image_id);
    if isKey(byName, nm), byName(nm) = [byName(nm), i]; else, byName(nm) = i; end
end
dirImgs = dir(fullfile(paths.hrsidImages, '*.png')); nAll = numel(dirImgs);
rng_ = q.Range(1):min(q.Range(2), nAll);
cP = cell(nAll, 1); cM = cell(nAll, 1); cX = cell(nAll, 1); cS = cell(nAll, 1);
imgNames = cell(nAll, 1); imgNShips = zeros(nAll, 1); imgNTrig = zeros(nAll, 1); imgNComp = zeros(nAll, 1);
t0 = tic;
for ii = rng_
    name = dirImgs(ii).name; imgNames{ii} = name;
    I = double(imread(fullfile(paths.hrsidImages, name))); if ndims(I) == 3, I = I(:,:,1); end
    h0 = floor(size(I,1)/2)*2; w0 = floor(size(I,2)/2)*2; I = I(1:h0,1:w0);
    h0 = floor(h0/f)*f; w0 = floor(w0/f)*f; I = I(1:h0,1:w0);
    idxs = []; if isKey(byName, name), idxs = byName(name); end
    nS = numel(idxs); imgNShips(ii) = nS;
    % ---- ship polygons at full resolution, dilated by TolPx
    dilM = cell(1, nS);
    for s = 1:nS
        a = anns(idxs(s)); b = a.bbox(:)';
        x1 = max(1, round(b(1))); y1 = max(1, round(b(2))); x2 = min(w0, round(b(1)+b(3))); y2 = min(h0, round(b(2)+b(4)));
        bm = false(h0, w0); bm(y1:y2, x1:x2) = true; sm = bm;
        try
            sg = a.segmentation; if iscell(sg), sg = sg{1}; end
            xy = reshape(double(sg(:)), 2, [])'; pm = poly2mask(xy(:,1) + 1, xy(:,2) + 1, h0, w0) & bm; if nnz(pm) >= 3, sm = pm; end
        catch
        end
        dilM{s} = imdilate(sm, strel('square', 2*q.TolPx + 1));
    end
    % ---- pooled-domain CFAR
    if f == 1, Id = I; else, Id = blockmean(I, f); end
    if strcmp(q.Mode, 'log') && f > 1, x = blockmean(log(sqrt(I + 0.5)), f); else, x = log(sqrt(Id + 0.5)); end
    [h, w] = size(x);
    [c1, c2] = ring_stats(x, Gp, TK, TG, X0);
    prm = WeibullCFAR_Params(c2);
    D = (x > c1 + WeibullCFAR_TLog(prm, q.Pfa)) & prm.valid;
    C = x - c1; Dg = D & (C >= q.Gate);
    switch q.Event
        case 'nms',  E = nms_trigger(D) & Dg;
        case 'peak3', Cd = -inf(h, w); Cd(Dg) = C(Dg); E = Dg & (Cd >= imdilate(Cd, ones(3)));
        case 'peak5', Cd = -inf(h, w); Cd(Dg) = C(Dg); E = Dg & (Cd >= imdilate(Cd, ones(5)));
        case 'comp'
            E = false(h, w); li = find(Dg);
            if ~isempty(li), lab = bwlabel(D, 8); lb = lab(li); [~, ord] = sort(C(li), 'descend'); [~, first] = unique(lb(ord), 'stable'); E(li(ord(first))) = true; end
    end
    [ey, ex] = find(E); nE = numel(ey); imgNTrig(ii) = nE;
    if nE == 0, continue; end
    yc = (ey - 1) * f + (f + 1) / 2; xc = (ex - 1) * f + (f + 1) / 2;       % full-resolution centre (1-based, fractional)
    ry = min(max(round(yc), 1), h0); rx = min(max(round(xc), 1), w0);
    gt = zeros(nE, 1); gt2 = zeros(nE, 1);                   % first / second ship whose dilated polygon contains the event (touching ships share events)
    for s = 1:nS
        hit = dilM{s}(sub2ind([h0 w0], ry, rx));
        gt2(hit & gt ~= 0 & gt2 == 0) = s; gt(hit & gt == 0) = s;
    end
    lin = sub2ind([h w], ey, ex);
    if q.StorePatches
        qc = floor(min(max((0.5*log(I + 0.5) - XLo) / (XHi - XLo), 0), 1) * 255 + 0.5);
        P = floor((qc(1:2:end,1:2:end) + qc(1:2:end,2:2:end) + qc(2:2:end,1:2:end) + qc(2:2:end,2:2:end) + 2) / 4);
        hp = h0/2; wp = w0/2;
        pat = zeros(2*HALF, 2*HALF, nE, 'uint8');
        for n = 1:nE
            j = floor((yc(n) - 1) / 2); i = floor((xc(n) - 1) / 2);            % 0-based pooled-2x2 position
            rr = min(max((j-HALF):(j+HALF-1), 0), hp-1) + 1; cc2 = min(max((i-HALF):(i+HALF-1), 0), wp-1) + 1;
            pat(:,:,n) = uint8(P(rr, cc2));
        end
        cP{ii} = pat;
    end
    if q.Ctx
        qc4 = floor(min(max((0.5*log(I + 0.5) - XLo) / (XHi - XLo), 0), 1) * 255 + 0.5);
        h4 = floor(h0/4); w4 = floor(w0/4);
        P4 = floor(squeeze(sum(sum(reshape(qc4(1:h4*4,1:w4*4), 4, h4, 4, w4), 1), 3)) / 16 + 0.5);     % 4x4-pooled QROM store (mean of 16 codes)
        ctx = zeros(2*HALF, 2*HALF, nE, 'uint8');
        for n = 1:nE
            j = floor((yc(n) - 1) / 4); i = floor((xc(n) - 1) / 4);
            rr = min(max((j-HALF):(j+HALF-1), 0), h4-1) + 1; cc4 = min(max((i-HALF):(i+HALF-1), 0), w4-1) + 1;
            ctx(:,:,n) = uint8(P4(rr, cc4));
        end
        cX{ii} = ctx;
        xi_ = log(sqrt(I + 0.5));
        imgst = [mean(xi_(:)), std(xi_(:)), mean(I(:) > 200), mean(I(:) < 5), log(1 + nE)];
        cS{ii} = [double(C(lin)), double(c1(lin)), sqrt(double(c2(lin))), double(x(lin)), repmat(imgst, nE, 1)];
    end
    cM{ii} = [double(gt > 0), double(gt), double(C(lin)), yc, xc, zeros(nE, 1), double(c1(lin)), double(x(lin)), repmat(ii, nE, 1), double(gt2)];
    if mod(ii - rng_(1) + 1, 100) == 0, fprintf('  [%d/%d] %.0fs  events so far %d\n', ii - rng_(1) + 1, numel(rng_), toc(t0), sum(cellfun(@(c) size(c,1), cM))); end
end
M = vertcat(cM{:});
labels = logical(M(:,1)); gtIdx = uint8(M(:,2)); gate = single(M(:,3)); tyx = single(M(:,4:5));
compArea = uint16(M(:,6)); c1 = single(M(:,7)); xt = single(M(:,8)); imgIdx = uint16(M(:,9)); gtIdx2 = uint8(M(:,10));
if q.StorePatches, patches = cat(3, cP{:}); else, patches = zeros(32,32,0,'uint8'); end
if q.Ctx, ctx = cat(3, cX{:}); side = single(vertcat(cS{:})); else, ctx = zeros(32,32,0,'uint8'); side = zeros(0, 9, 'single'); end
meta = struct('Pool', f, 'Mode', q.Mode, 'Sli', q.Sli, 'Guard', q.Guard, 'Pfa', q.Pfa, 'Gate', q.Gate, 'Event', q.Event, 'TolPx', q.TolPx, ...
    'Spec', 'pooled-domain prescreen, padded border, patch = 32x32 of the pooled 2x2 QROM store', 'Range', q.Range);
fprintf('Total %d events (%d on ships = %.2f%%)\n', numel(labels), sum(labels), 100*mean(labels));
save(fullfile(paths.results, q.OutName), 'patches', 'labels', 'imgIdx', 'gtIdx', 'gate', 'tyx', 'compArea', 'c1', 'xt', 'gtIdx2', 'ctx', 'side', 'imgNames', 'imgNShips', 'imgNTrig', 'imgNComp', 'meta', '-v7.3');
fprintf('Wrote %s (%.0fs)\n', q.OutName, toc(t0));
end

function [c1, c2] = ring_stats(x, Gp, TK, TG, X0)
y = x - X0; Wp = padarray(ones(size(x)), [TK TK], 'symmetric'); Yp = padarray(y, [TK TK], 'symmetric'); Y2p = padarray(y.^2, [TK TK], 'symmetric');
N = ring(Wp, Gp.sli, Gp.guard, TK, TG); S1 = ring(Yp, Gp.sli, Gp.guard, TK, TG); S2 = ring(Y2p, Gp.sli, Gp.guard, TK, TG);
c1 = X0 + S1 ./ N; c2 = max((S2 - S1.^2 ./ N) ./ (N - 1), 1e-9);
end
function S = ring(Ap, sli, guard, TK, TG)
big = boxsum_valid(Ap, sli); Ag = Ap(TK-TG+1:end-(TK-TG), TK-TG+1:end-(TK-TG)); S = big - boxsum_valid(Ag, guard);
end
function B = boxsum_valid(A, k)
C = cumsum(cumsum(A, 1), 2); C = padarray(C, [1 1], 0, 'pre');
B = C(k+1:end, k+1:end) - C(1:end-k, k+1:end) - C(k+1:end, 1:end-k) + C(1:end-k, 1:end-k);
end
function B = blockmean(A, f)
[h, w] = size(A); h = floor(h/f)*f; w = floor(w/f)*f; B = squeeze(mean(mean(reshape(A(1:h,1:w), f, h/f, f, w/f), 1), 3));
end
function T = nms_trigger(D)
[h, w] = size(D); up = [false(1,w); D(1:end-1,:)]; prvL = [false(h,1), D(:,1:end-1)]; upL = [false(h,1), up(:,1:end-1)]; upR = [up(:,2:end), false(h,1)];
T = D & ~prvL & ~up & ~upL & ~upR;
end
