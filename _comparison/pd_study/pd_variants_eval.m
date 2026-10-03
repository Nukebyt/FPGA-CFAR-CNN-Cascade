function pd_variants_eval(varargin)
%PD_VARIANTS_EVAL  Phase 2-4 harness: strict prescreen Pd and CNN workload for many prescreen variants, per image (Paper 2, software only).
%
%   Variant = front-end GROUP x Pfa x gate tau x event definition.
%   GROUP: window (sli, guard), domain (full resolution | 2x2-pooled), reference-ring estimator (plain | censored), border (valid | padded).
%     * pooled domain : intensity averaged over 2x2 before log/CFAR (multi-look); a ship is half as large relative to the same window
%     * censored      : two-pass iterative censoring -- pass 1 detects at Pfa CENS_PFA, detections (dilated 5x5) are excluded from the reference ring,
%                       pass 2 re-estimates c1/c2 from the remaining cells (cells of the ship itself / neighbours no longer inflate the background)
%     * padded border : symmetric padding by the half window, so the outer pixels are evaluated (the RTL leaves the outer 8 px unevaluated)
%   EVENT definitions (each also needs x - c1 >= gate):
%     nms  : the streaming NMS corner of every detection run (current RTL)
%     peak : local maximum of the contrast x - c1 in a 5x5 neighbourhood among detected+gated pixels (robust when runs merge at high Pfa)
%     comp : one event per 8-connected detection component, at its highest-contrast gated pixel (software upper bound, not streaming-friendly)
%   Per image and variant it stores 7 counts (R, last dim):
%     [nShips, hitMask (detected pixel on the polygon), hitEv (event on the polygon +-3 px full-res / +-2 px pooled), nEvents, nEventsOffShip,
%      falsePix (detected pixels outside every dilated ship polygon), bgPix (pixels outside every dilated ship polygon)]
%   usage: pd_variants_eval('ListFile','pd_variants/img_tune.txt','Range',[1 200],'OutName','pd_variants/tune_p1.mat')

p = inputParser;
addParameter(p, 'ListFile', 'pd_variants/img_tune.txt'); addParameter(p, 'Range', [1 inf]); addParameter(p, 'OutName', 'pd_variants/tune_p1.mat');
addParameter(p, 'GroupSet', 'all');
parse(p, varargin{:});
paths = cfar_setup();
lst = round(load(fullfile(paths.results, p.Results.ListFile)));
r = p.Results.Range; lst = lst(r(1):min(r(2), numel(lst)));
X0 = 1.2125; CENS_PFA = 3e-3;
PFAS = [1e-3 3e-3 1e-2 3e-2]; GATES = [0.5 0.75 1.0]; EVTS = {'nms', 'peak', 'comp'};

% ---- groups: name, sli, guard, pool, censor, border
G = struct('name', {}, 'sli', {}, 'guard', {}, 'pool', {}, 'cens', {}, 'pad', {});
add = @(sli, g, pool, cens, pad) struct('name', sprintf('%s%d_%d%s%s', ternary(pool==2,'P','F'), sli, g, ternary(cens,'c',''), ternary(pad,'b','v')), ...
      'sli', sli, 'guard', g, 'pool', pool, 'cens', cens, 'pad', pad);
G(end+1) = add(17, 13, 1, false, false);                        % baseline = the RTL's window and border
G(end+1) = add(17, 13, 1, false, true);                         % + padded border
for w = [21 13; 25 17; 33 21; 41 27; 49 33]', G(end+1) = add(w(1), w(2), 1, false, true); end %#ok<AGROW>
for w = [17 13; 21 13; 25 17; 33 21]', G(end+1) = add(w(1), w(2), 1, true, true); end %#ok<AGROW>
for w = [9 7; 13 9; 17 13; 25 17]', G(end+1) = add(w(1), w(2), 2, false, true); end %#ok<AGROW>
for w = [13 9; 17 13]', G(end+1) = add(w(1), w(2), 2, true, true); end %#ok<AGROW>
nG = numel(G); nP = numel(PFAS); nGt = numel(GATES); nE = numel(EVTS);

raw = jsondecode(fileread(paths.hrsidAnnFile));
id2name = containers.Map('KeyType', 'double', 'ValueType', 'char');
for i = 1:numel(raw.images), id2name(raw.images(i).id) = raw.images(i).file_name; end
byName = containers.Map('KeyType', 'char', 'ValueType', 'any'); anns = raw.annotations;
for i = 1:numel(anns)
    nm = id2name(anns(i).image_id);
    if isKey(byName, nm), byName(nm) = [byName(nm), i]; else, byName(nm) = i; end
end
dirImgs = dir(fullfile(paths.hrsidImages, '*.png'));

nImg = numel(lst);
R = zeros(nImg, nG, nP, nGt, nE, 7, 'single');
t0 = tic;
for ii = 1:nImg
    name = dirImgs(lst(ii)).name;
    I = double(imread(fullfile(paths.hrsidImages, name))); if ndims(I) == 3, I = I(:,:,1); end
    h0 = floor(size(I,1)/2)*2; w0 = floor(size(I,2)/2)*2; I = I(1:h0,1:w0);
    idxs = []; if isKey(byName, name), idxs = byName(name); end
    nS = numel(idxs); shipM = cell(1, nS);
    for s = 1:nS
        a = anns(idxs(s)); b = a.bbox(:)';
        x1 = max(1, round(b(1))); y1 = max(1, round(b(2))); x2 = min(w0, round(b(1)+b(3))); y2 = min(h0, round(b(2)+b(4)));
        bm = false(h0, w0); bm(y1:y2, x1:x2) = true; sm = bm;
        try
            sg = a.segmentation; if iscell(sg), sg = sg{1}; end
            xy = reshape(double(sg(:)), 2, [])'; pm = poly2mask(xy(:,1) + 1, xy(:,2) + 1, h0, w0) & bm; if nnz(pm) >= 3, sm = pm; end
        catch
        end
        shipM{s} = sm;
    end
    for pool = 1:2
        if pool == 1, Id = I; sm_d = shipM; dil = 3; else
            Id = (I(1:2:end,1:2:end) + I(1:2:end,2:2:end) + I(2:2:end,1:2:end) + I(2:2:end,2:2:end)) / 4;
            sm_d = cellfun(@(m) m(1:2:end,1:2:end) | m(1:2:end,2:2:end) | m(2:2:end,1:2:end) | m(2:2:end,2:2:end), shipM, 'UniformOutput', false); dil = 2;
        end
        [h, w] = size(Id); x = log(sqrt(Id + 0.5));
        % pixel -> ship index lists (mask, and dilated mask), vectorised hit tests
        idxM = []; sidM = []; idxD = []; sidD = []; Mdil = false(h, w);
        for s = 1:nS
            dm = imdilate(sm_d{s}, strel('square', 2*dil + 1)); Mdil = Mdil | dm;
            li = find(sm_d{s}); idxM = [idxM; li]; sidM = [sidM; repmat(s, numel(li), 1)]; %#ok<AGROW>
            li = find(dm);      idxD = [idxD; li]; sidD = [sidD; repmat(s, numel(li), 1)]; %#ok<AGROW>
        end
        bgPix = nnz(~Mdil);
        for g = find([G.pool] == pool)
            Gp = G(g); TK = (Gp.sli - 1) / 2; TG = (Gp.guard - 1) / 2;
            [c1, c2, valid] = ring_stats(x, Gp, TK, TG, X0, []);
            prm = WeibullCFAR_Params(c2);
            if Gp.cens
                T1 = c1 + WeibullCFAR_TLog(prm, CENS_PFA); D1 = (x > T1) & prm.valid;
                Mc = imdilate(D1, strel('square', 5));
                [c1, c2, valid] = ring_stats(x, Gp, TK, TG, X0, ~Mc);
                prm = WeibullCFAR_Params(c2);
            end
            C = x - c1; vmask = valid & prm.valid;
            for ip = 1:nP
                D = (x > c1 + WeibullCFAR_TLog(prm, PFAS(ip))) & vmask;
                lab = []; if any(strcmp(EVTS, 'comp')), lab = bwlabel(D, 8); end
                hm = hits(D, idxM, sidM, nS);
                fp = nnz(D & ~Mdil);
                for ig = 1:nGt
                    Dg = D & (C >= GATES(ig));
                    for ie = 1:nE
                        switch EVTS{ie}
                            case 'nms',  E = nms_trigger(D) & Dg;
                            case 'peak'
                                Cd = -inf(h, w); Cd(Dg) = C(Dg);
                                E = Dg & (Cd >= imdilate(Cd, ones(5)));
                            case 'comp'
                                E = false(h, w); li = find(Dg);
                                if ~isempty(li)
                                    lb = lab(li); [~, ord] = sort(C(li), 'descend'); [~, first] = unique(lb(ord), 'stable'); E(li(ord(first))) = true;
                                end
                        end
                        he = hits(E, idxD, sidD, nS);
                        R(ii, g, ip, ig, ie, :) = single([nS, sum(hm), sum(he), nnz(E), nnz(E & ~Mdil), fp * (1 + 3*(pool==2)), bgPix * (1 + 3*(pool==2))]);
                    end
                end
            end
        end
    end
    if mod(ii, 20) == 0, fprintf('  [%d/%d] %.0fs\n', ii, nImg, toc(t0)); end
end
meta = struct('groups', {{G.name}}, 'pfas', PFAS, 'gates', GATES, 'events', {EVTS}, 'list', lst, 'cens_pfa', CENS_PFA, ...
    'counts', {{'nShips','hitMask','hitEv','nEvents','nEventsOffShip','falsePix','bgPix'}});
outf = fullfile(paths.results, p.Results.OutName);
save(outf, 'R', 'meta', '-v7');
fprintf('wrote %s (%.0fs)\n', outf, toc(t0));
end

function [c1, c2, valid] = ring_stats(x, Gp, TK, TG, X0, wmap)
% mean / unbiased variance of x over the reference ring (sli x sli minus guard x guard); optional weights wmap (1 = use cell, 0 = censored)
[h, w] = size(x);
y = x - X0;
if isempty(wmap), wmap = true(h, w); end
wm = double(wmap);
if Gp.pad
    pad = @(A) padarray(A, [TK TK], 'symmetric');
else
    pad = @(A) padarray(A, [TK TK], 0);
end
Wp = pad(wm); Yp = pad(y .* wm); Y2p = pad(y.^2 .* wm);
N  = ring(Wp, Gp.sli, Gp.guard, TK, TG);
S1 = ring(Yp, Gp.sli, Gp.guard, TK, TG);
S2 = ring(Y2p, Gp.sli, Gp.guard, TK, TG);
ok = N >= max(8, 0.35 * (Gp.sli^2 - Gp.guard^2));
Nf = max(N, 2);
c1 = X0 + S1 ./ Nf;
c2 = max((S2 - S1.^2 ./ Nf) ./ (Nf - 1), 1e-9);
valid = ok;
if ~Gp.pad
    v = false(h, w); v(TK+1:end-TK, TK+1:end-TK) = true; valid = valid & v;
end
end

function S = ring(Ap, sli, guard, TK, TG)
big = boxsum_valid(Ap, sli);
Ag = Ap(TK-TG+1:end-(TK-TG), TK-TG+1:end-(TK-TG));
S = big - boxsum_valid(Ag, guard);
end

function B = boxsum_valid(A, k)
C = cumsum(cumsum(A, 1), 2);
C = padarray(C, [1 1], 0, 'pre');
B = C(k+1:end, k+1:end) - C(1:end-k, k+1:end) - C(k+1:end, 1:end-k) + C(1:end-k, 1:end-k);
end

function hv = hits(Mp, idx, sid, nS)
if nS == 0, hv = zeros(0, 1); return; end
hv = accumarray(sid, double(Mp(idx)), [nS 1], @max, 0);
end

function T = nms_trigger(D)
[h, w] = size(D);
up   = [false(1,w); D(1:end-1,:)];
prvL = [false(h,1), D(:,1:end-1)];
upL  = [false(h,1), up(:,1:end-1)];
upR  = [up(:,2:end), false(h,1)];
T = D & ~prvL & ~up & ~upL & ~upR;
end

function v = ternary(c, a, b)
if c, v = a; else, v = b; end
end
