function pd_study_extract(varargin)
%PD_STUDY_EXTRACT  Per-ship and per-image diagnostics of the Weibull prescreen on HRSID (Paper 2 "raise the prescreen Pd" study, phase 1).
%
%   Why a ship is missed, measured per ship.  For every annotated ship (HRSID polygon mask, box as fallback) at every Pfa level:
%     hit_k      any detected pixel inside the GT box           (the metrics.py / computePdPfa definition)
%     margin_k   max over the box of  x - T_k                    (log-amplitude units; >0 = detected)
%   and the exact decomposition of the gap at the best ship pixel for the 1e-3 plane
%     T - x_best = (c1 - mu_bg)  +  delta  +  (mu_bg - x_best)
%                   window/mean     Weibull        true contrast
%                   inflation       threshold      deficit of the ship against the local sea
%   plus ship geometry (size, distance to the image border / the CFAR valid region, neighbours), the local background
%   (mean/std in log domain, heterogeneity), the scene class is merged later (HRSID inshore/offshore lists).
%   Per image: detected / false pixels, NMS triggers and gated (x-c1>=0.75) triggers at every Pfa level, clutter descriptors.
%
%   Float model (cfar_front_end + WeibullCFAR_TLog) so ANY Pfa can be evaluated; the bit-exact fixed-point hit flag at the 1e-3 plane is
%   also stored (hitFx) to show that the float analysis describes the hardware.
%
%   usage:  pd_study_extract('Range',[1 5604],'OutName','pd_study/part1.mat')

p = inputParser;
addParameter(p, 'Range', [1 5604]); addParameter(p, 'OutName', 'pd_study/part1.mat');
addParameter(p, 'Sli', 17); addParameter(p, 'Guard', 13);
parse(p, varargin{:});
paths = cfar_setup();
sli = p.Results.Sli; guard = p.Results.Guard; TK = (sli-1)/2;
PFAS = [1e-1 3e-2 1e-2 3e-3 1e-3 1e-4 1e-5 1e-6];  nP = numel(PFAS);  P1 = find(PFAS == 1e-3);
RING = 20;                                              % background ring width around a ship box (px)

% ---- annotations (boxes + polygons)
raw = jsondecode(fileread(paths.hrsidAnnFile));
imgsInfo = raw.images; anns = raw.annotations;
id2name = containers.Map('KeyType', 'double', 'ValueType', 'char');
for i = 1:numel(imgsInfo), id2name(imgsInfo(i).id) = imgsInfo(i).file_name; end
byName = containers.Map('KeyType', 'char', 'ValueType', 'any');
for i = 1:numel(anns)
    nm = id2name(anns(i).image_id);
    if isKey(byName, nm), byName(nm) = [byName(nm), i]; else, byName(nm) = i; end
end
dirImgs = dir(fullfile(paths.hrsidImages, '*.png'));

% ---- fixed-point model (hardware plane 1e-3) for the cross-check flag
cfgFx = fixedpoint_config(sli, guard);
sharedDir = fullfile(paths.root, 'lut', 'shared');
wl = load(fullfile(paths.root, 'lut', 'weibull', 'weibull_delta_lut.mat'));
nlut = wl.entries_per_pfa;

Sn = {'img','annIdx','bx1','by1','bx2','by2','bw','bh','areaMask','validFrac','distBorder','xmax','xp90','xmean','imax', ...
      'muBg','sdBg','ampBg','nBg','dxMax','zMax','c1Best','c2Best','deltaBest','xBest','gapInfl','gapDelta','gapContrast', ...
      'nbrDist','nShips','nHitPix','hitFx'};
for k = 1:nP, Sn{end+1} = sprintf('hit%d', k); Sn{end+1} = sprintf('margin%d', k); end %#ok<AGROW>
% v2 columns: clean-sea counterfactual, reference-ring contamination, strict hit definitions (float planes 1..8, hardware-exact planes 1..4)
Sn = [Sn, {'deltaClean','marginClean','c2Bg','ringOwn','ringOther','ringBright'}];
for k = 1:nP, Sn{end+1} = sprintf('maskHit%d', k); Sn{end+1} = sprintf('evComp%d', k); Sn{end+1} = sprintf('evMask%d', k); end %#ok<AGROW>
for q = 1:4, Sn{end+1} = sprintf('maskHitFx%d', q); Sn{end+1} = sprintf('evCompFx%d', q); Sn{end+1} = sprintf('evMaskFx%d', q); Sn{end+1} = sprintf('hitFx%d', q); end %#ok<AGROW>
Imn = {'img','h','w','nShips','mx','sx','pBright','pDark','meanC2','validPix'};
for k = 1:nP
    Imn{end+1} = sprintf('detPix%d', k); Imn{end+1} = sprintf('falsePix%d', k); Imn{end+1} = sprintf('bgPix%d', k);
    Imn{end+1} = sprintf('trig%d', k);   Imn{end+1} = sprintf('gated%d', k); %#ok<AGROW>
end
S = zeros(0, numel(Sn)); Im = zeros(numel(Imn) > 0, numel(Imn)); Im = zeros(0, numel(Imn));

t0 = tic; rng_ = p.Results.Range(1):min(p.Results.Range(2), numel(dirImgs));
for ii = rng_
    name = dirImgs(ii).name;
    I = double(imread(fullfile(paths.hrsidImages, name)));
    if ndims(I) == 3, I = I(:,:,1); end
    h = floor(size(I,1)/2)*2; w = floor(size(I,2)/2)*2; I = I(1:h,1:w);
    fe = cfar_front_end(I, sli, guard); prm = WeibullCFAR_Params(fe.c2);
    x = fe.x; c1 = fe.c1; valid = prm.valid & true(h, w);
    % fixed-point plane-1e-3 detection map (bit-exact RTL arithmetic)
    fx = cfar_front_end_fixed(I, sli, guard, sharedDir, 'Config', cfgFx);
    aidx = min(max(round((fx.c2(:) - wl.addr_min) / (wl.addr_max - wl.addr_min) * (nlut-1)), 0), nlut-1);
    c1c = double(fx.c1_code(:)); c1t = sign(c1c) .* floor((abs(c1c) + 4) / 8);
    vfx = false(h, w); vfx(TK+1:end-TK, TK+1:end-TK) = true;
    Dfx = cell(1, 4);
    for q = 1:4
        dn = double(wl.code((q-1)*nlut + aidx + 1)); dc = sign(dn) .* floor((abs(dn) + 2) / 4);
        Dq = reshape(double(fx.x_code(:)) > (c1t + dc - 1242) * 16, [h w]) & vfx; Dq(TK+1, TK+1) = false; Dfx{q} = Dq;
    end
    Gfx = reshape(double(fx.x_code(:)) + 19866 - 2*double(fx.c1_code(:)), [h w]) >= 12288;      % hardware gate x - c1 >= 0.75 (Q.14)

    T = cell(1, nP); D = cell(1, nP); delta = cell(1, nP);
    for k = 1:nP
        delta{k} = WeibullCFAR_TLog(prm, PFAS(k));
        T{k} = c1 + delta{k}; D{k} = (x > T{k}) & valid;
    end

    % ---- candidate events (gated NMS triggers) and component labels, float planes 1..nP and hardware planes 1..4
    nms = @(Dm) nms_trigger(Dm);
    Gfl = (x - c1) >= 0.75;
    Lf = cell(1, nP); Tgf = cell(1, nP); CTf = cell(1, nP);
    for k = 1:nP
        Lf{k} = bwlabel(D{k}, 8); Tgf{k} = nms(D{k}) & Gfl;
        ct = false(max(Lf{k}(:)) + 1, 1); ids = unique(Lf{k}(Tgf{k})); ct(ids + 1) = true; CTf{k} = ct;   % label id (+1) -> has a gated trigger
    end
    Lx = cell(1, 4); Tgx = cell(1, 4); CTx = cell(1, 4);
    for q = 1:4
        Lx{q} = bwlabel(Dfx{q}, 8); Tgx{q} = nms(Dfx{q}) & Gfx;
        ct = false(max(Lx{q}(:)) + 1, 1); ids = unique(Lx{q}(Tgx{q})); ct(ids + 1) = true; CTx{q} = ct;
    end

    % ---- ground truth
    idxs = []; if isKey(byName, name), idxs = byName(name); end
    nS = numel(idxs);
    boxM = cell(1, nS); shipM = cell(1, nS); bb = zeros(nS, 4);
    allBox = false(h, w);
    for s = 1:nS
        a = anns(idxs(s)); b = a.bbox(:)';
        x1 = max(1, round(b(1))); y1 = max(1, round(b(2))); x2 = min(w, round(b(1)+b(3))); y2 = min(h, round(b(2)+b(4)));
        bm = false(h, w); bm(y1:y2, x1:x2) = true; boxM{s} = bm; bb(s,:) = [x1 y1 x2 y2]; allBox = allBox | bm;
        sm = bm;
        try
            sg = a.segmentation; if iscell(sg), sg = sg{1}; end
            xy = reshape(double(sg(:)), 2, [])';
            pm = poly2mask(xy(:,1) + 1, xy(:,2) + 1, h, w) & bm;
            if nnz(pm) >= 3, sm = pm; end
        catch
        end
        shipM{s} = sm;
    end
    allShip = false(h, w); for s_ = 1:nS, allShip = allShip | shipM{s_}; end
    nearShip = imdilate(allBox, strel('square', 7));         % exclude ship + 3 px margin from background estimates

    xi = x(~allBox & ~isnan(x));
    Imrow = nan(1, numel(Imn));
    Imrow(1:10) = [ii h w nS mean(xi) std(xi) mean(I(:) > 200) mean(I(:) < 5) mean(fe.c2(valid)) nnz(valid)];
    for k = 1:nP
        up   = [false(1,w); D{k}(1:end-1,:)];
        prvL = [false(h,1), D{k}(:,1:end-1)];
        upL  = [false(h,1), up(:,1:end-1)];
        upR  = [up(:,2:end), false(h,1)];
        Tm = D{k} & ~prvL & ~up & ~upL & ~upR;
        gate = (x - c1) >= 0.75;
        o = 10 + (k-1)*5;
        Imrow(o+1:o+5) = [nnz(D{k}), nnz(D{k} & ~allBox), nnz(~allBox), nnz(Tm), nnz(Tm & gate)];
    end
    Im(end+1, :) = Imrow; %#ok<AGROW>

    for s = 1:nS
        sm = shipM{s}; bm = boxM{s}; r = bb(s,:);
        row = nan(1, numel(Sn));
        xs = x(sm); xs = xs(~isnan(xs));
        rr = max(1, r(2)-RING):min(h, r(4)+RING); cc = max(1, r(1)-RING):min(w, r(3)+RING);
        win = false(h, w); win(rr, cc) = true;
        bg = win & ~nearShip;
        xb = x(bg); xb = xb(~isnan(xb));
        if numel(xb) < 20, xb = xi; end
        muBg = mean(xb); sdBg = std(xb); ampBg = mean(sqrt(I(bg) + 0.5));
        % distance of the box to the image border and to the CFAR valid interior; fraction of the box that CAN be detected
        distBorder = min([r(1)-1, r(2)-1, w-r(3), h-r(4)]);
        validFrac = mean(valid(bm));
        % best ship pixel for the 1e-3 plane (among CFAR-valid pixels of the mask)
        mk = sm & valid; if ~any(mk(:)), mk = bm & valid; end
        if any(mk(:))
            m1 = x - T{P1}; m1(~mk) = -inf; [mb, lin] = max(m1(:));
            xBest = x(lin); c1Best = c1(lin); c2Best = fe.c2(lin); dBest = delta{P1}(lin);
            gapInfl = c1Best - muBg; gapDelta = dBest; gapContrast = muBg - xBest;
        else
            xBest = NaN; c1Best = NaN; c2Best = NaN; dBest = NaN; gapInfl = NaN; gapDelta = NaN; gapContrast = NaN; mb = -inf;
        end
        % neighbours
        nd = inf;
        for q = 1:nS
            if q == s, continue; end
            dx = max([bb(q,1) - r(3), r(1) - bb(q,3), 0]); dy = max([bb(q,2) - r(4), r(2) - bb(q,4), 0]);
            nd = min(nd, hypot(dx, dy));
        end
        xmaxS = max(xs); if isempty(xmaxS), xmaxS = NaN; end
        row(1:end) = NaN;
        row(1:32) = [ii idxs(s) r(1) r(2) r(3) r(4) r(3)-r(1)+1 r(4)-r(2)+1 nnz(sm) validFrac distBorder ...
            xmaxS prctile_safe(xs, 90) mean(xs) max(I(sm)) muBg sdBg ampBg numel(xb) xmaxS-muBg (xmaxS-muBg)/max(sdBg,1e-6) ...
            c1Best c2Best dBest xBest gapInfl gapDelta gapContrast nd nS nnz(D{P1} & bm) any(Dfx{1}(bm))];
        for k = 1:nP
            mk_ = bm & valid; mm = x - T{k}; mm(~mk_) = -inf;
            row(32 + 2*(k-1) + 1) = any(D{k}(bm));
            row(32 + 2*(k-1) + 2) = max(mm(:));
        end
        % ---- v2: clean-sea counterfactual (c1, c2 from the ship-free sea around the ship, same Weibull offset rule)
        c2Bg = var(xb); deltaClean = NaN; marginClean = NaN;
        try
            prmC = WeibullCFAR_Params(c2Bg); deltaClean = double(WeibullCFAR_TLog(prmC, 1e-3)); marginClean = xmaxS - (muBg + deltaClean);
        catch
        end
        % ---- reference-ring contamination at the best ship pixel (SLI 17 / guard 13 ring = 17x17 minus 13x13)
        ringOwn = NaN; ringOther = NaN; ringBright = NaN;
        if any(mk(:))
            [br, bc] = ind2sub([h w], lin);
            rr_ = max(1, br-TK):min(h, br+TK); cc_ = max(1, bc-TK):min(w, bc+TK);
            [CC, RR] = meshgrid(cc_, rr_); ringSel = ~(abs(RR - br) <= (guard-1)/2 & abs(CC - bc) <= (guard-1)/2);
            li = sub2ind([h w], RR(ringSel), CC(ringSel));
            inOwn = sm(li); inAny = allShip(li); xr = x(li);
            ringOwn = mean(inOwn); ringOther = mean(inAny & ~inOwn); ringBright = mean(~inAny & xr > muBg + 2*sdBg);
        end
        row(33 + 2*nP : 32 + 2*nP + 6) = [deltaClean marginClean c2Bg ringOwn ringOther ringBright];
        be = imdilate(sm, strel('square', 7));      % ship polygon + 3 px: a gated trigger here is an event ON the ship (what the CNN must classify as ship)
        o = 32 + 2*nP + 6;
        for k = 1:nP
            ids = unique(Lf{k}(bm)); ids = ids(ids > 0);
            row(o + 3*(k-1) + 1) = any(D{k}(sm));
            row(o + 3*(k-1) + 2) = any(CTf{k}(ids + 1));
            row(o + 3*(k-1) + 3) = any(Tgf{k}(be));
        end
        o = o + 3*nP;
        for q = 1:4
            ids = unique(Lx{q}(bm)); ids = ids(ids > 0);
            row(o + 4*(q-1) + 1) = any(Dfx{q}(sm));
            row(o + 4*(q-1) + 2) = any(CTx{q}(ids + 1));
            row(o + 4*(q-1) + 3) = any(Tgx{q}(be));
            row(o + 4*(q-1) + 4) = any(Dfx{q}(bm));
        end
        S(end+1, :) = row; %#ok<AGROW>
    end
    if mod(ii - rng_(1) + 1, 100) == 0
        fprintf('  [%d/%d] %.0fs  ships so far %d\n', ii - rng_(1) + 1, numel(rng_), toc(t0), size(S,1));
    end
end
meta = struct('Sli', sli, 'Guard', guard, 'PFAS', PFAS, 'RING', RING, 'range', p.Results.Range);
outf = fullfile(paths.results, p.Results.OutName);
save(outf, 'S', 'Sn', 'Im', 'Imn', 'meta', '-v7');
fprintf('wrote %s (%.0fs): %d ships, %d images\n', outf, toc(t0), size(S,1), size(Im,1));
end

function v = prctile_safe(a, q)
if isempty(a), v = NaN; return; end
a = sort(a(:)); v = a(max(1, min(numel(a), ceil(q/100*numel(a)))));
end

function T = nms_trigger(D)
% streaming NMS trigger: D & ~D(y,x-1) & ~D(y-1,x-1) & ~D(y-1,x) & ~D(y-1,x+1)
[h, w] = size(D);
up   = [false(1,w); D(1:end-1,:)];
prvL = [false(h,1), D(:,1:end-1)];
upL  = [false(h,1), up(:,1:end-1)];
upR  = [up(:,2:end), false(h,1)];
T = D & ~prvL & ~up & ~upL & ~upR;
end
