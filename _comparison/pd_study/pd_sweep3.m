function pd_sweep3(varargin)
%PD_SWEEP3  Whole-data-set Weibull-only sweep at Pfa 1e-2, 1e-3, 1e-4 for TWO prescreens (Paper 2):
%     group 1 = pooled 2x2 log-mean, window 17/13, padded border      group 2 = full resolution, window 17/13, padded border
%   gate x - c1 >= 0.75, 5x5 contrast-peak events, 4 px event tolerance.  Variant order v = (group-1)*3 + (1..3 <-> Pfa 1e-2, 1e-3, 1e-4).
%   Per SHIP : [img, rank, then for every variant: minDist (px, full-res, min distance from any event to the ship polygon; Inf = no event), maskHit]
%   Per IMAGE: [img, nShips, bgPix, then for every variant: nEvents, nEventsOffShip (> 4 px from every ship), falsePix (detected px > 4 px from every ship, full-res
%              equivalent), detPix (all detected px, full-res equivalent)]
%   usage: pd_sweep3('Range',[1 2800],'OutName','pd_sweep3/part1.mat')
p = inputParser; addParameter(p, 'Range', [1 5604]); addParameter(p, 'OutName', 'pd_sweep3/part1.mat'); parse(p, varargin{:});
paths = cfar_setup(); X0 = 1.2125; PFAS = [1e-2 1e-3 1e-4]; GATE = 0.75; TOL = 4;
GR = struct('pool', {2, 1}, 'log', {true, false}, 'sli', {17, 17}, 'guard', {13, 13}); nV = 6;
raw = jsondecode(fileread(paths.hrsidAnnFile));
id2name = containers.Map('KeyType', 'double', 'ValueType', 'char');
for i = 1:numel(raw.images), id2name(raw.images(i).id) = raw.images(i).file_name; end
byName = containers.Map('KeyType', 'char', 'ValueType', 'any'); anns = raw.annotations;
for i = 1:numel(anns), nm = id2name(anns(i).image_id); if isKey(byName, nm), byName(nm) = [byName(nm), i]; else, byName(nm) = i; end, end
dirImgs = dir(fullfile(paths.hrsidImages, '*.png'));
rng_ = p.Results.Range(1):min(p.Results.Range(2), numel(dirImgs));
S = zeros(0, 2 + 2*nV); Im = zeros(numel(rng_), 3 + 4*nV); t0 = tic; row = 0;
for ii = rng_
    row = row + 1; name = dirImgs(ii).name;
    I = double(imread(fullfile(paths.hrsidImages, name))); if ndims(I) == 3, I = I(:,:,1); end
    h0 = floor(size(I,1)/2)*2; w0 = floor(size(I,2)/2)*2; I = I(1:h0,1:w0);
    idxs = []; if isKey(byName, name), idxs = byName(name); end
    nS = numel(idxs); sm = cell(1, nS); Ds = cell(1, nS); Mun = false(h0, w0);
    for s = 1:nS
        a = anns(idxs(s)); b = a.bbox(:)';
        x1 = max(1, round(b(1))); y1 = max(1, round(b(2))); x2 = min(w0, round(b(1)+b(3))); y2 = min(h0, round(b(2)+b(4)));
        bm = false(h0, w0); bm(y1:y2, x1:x2) = true; m = bm;
        try
            sg = a.segmentation; if iscell(sg), sg = sg{1}; end
            xy = reshape(double(sg(:)), 2, [])'; pm = poly2mask(xy(:,1) + 1, xy(:,2) + 1, h0, w0) & bm; if nnz(pm) >= 3, m = pm; end
        catch
        end
        sm{s} = m; Mun = Mun | m; Ds{s} = bwdist(m);
    end
    if nS > 0, Du = bwdist(Mun); else, Du = inf(h0, w0); end
    near = Du <= TOL; bgPix = nnz(~near);
    shipRows = nan(nS, 2*nV); imRow = nan(1, 4*nV);
    for g = 1:2
        f = GR(g).pool; TK = (GR(g).sli - 1) / 2; TG = (GR(g).guard - 1) / 2;
        if f == 1, Id = I; x = log(sqrt(Id + 0.5)); smd = sm; nearD = near;
        else
            Id = blockmean(I, f); if GR(g).log, x = blockmean(log(sqrt(I + 0.5)), f); else, x = log(sqrt(Id + 0.5)); end
            smd = cellfun(@(m) blockany(m, f), sm, 'UniformOutput', false); nearD = blockany(near, f);
        end
        [h, w] = size(x);
        [c1, c2] = ring_stats(x, GR(g).sli, GR(g).guard, TK, TG, X0); prm = WeibullCFAR_Params(c2); C = x - c1;
        for ip = 1:3
            v = (g-1)*3 + ip;
            D = (x > c1 + WeibullCFAR_TLog(prm, PFAS(ip))) & prm.valid; Dg = D & (C >= GATE);
            Cd = -inf(h, w); Cd(Dg) = C(Dg); E = Dg & (Cd >= imdilate(Cd, ones(5)));
            [ey, ex] = find(E); yc = (ey - 1) * f + (f + 1) / 2; xc = (ex - 1) * f + (f + 1) / 2;
            ry = min(max(round(yc), 1), h0); rx = min(max(round(xc), 1), w0); lin = sub2ind([h0 w0], ry, rx);
            for s = 1:nS
                md = inf; if ~isempty(lin), md = min(Ds{s}(lin)); end
                shipRows(s, 2*(v-1) + 1) = md; shipRows(s, 2*(v-1) + 2) = any(D(smd{s}(:)));
            end
            nOff = nnz(Du(lin) > TOL);
            imRow(4*(v-1) + (1:4)) = [numel(ey), nOff, nnz(D & ~nearD) * f^2, nnz(D) * f^2];
        end
    end
    for s = 1:nS, S(end+1, :) = [ii, s, shipRows(s, :)]; end %#ok<AGROW>
    Im(row, :) = [ii, nS, bgPix, imRow];
    if mod(row, 100) == 0, fprintf('  [%d/%d] %.0fs  ships %d\n', row, numel(rng_), toc(t0), size(S, 1)); end
end
meta = struct('PFAS', PFAS, 'gate', GATE, 'tol', TOL, 'groups', {{'pooled2-log 17/13 padded', 'full-res 17/13 padded'}}, 'range', p.Results.Range);
save(fullfile(paths.results, p.Results.OutName), 'S', 'Im', 'meta', '-v7');
fprintf('wrote %s (%.0fs): %d ships, %d images\n', p.Results.OutName, toc(t0), size(S, 1), size(Im, 1));
end

function [c1, c2] = ring_stats(x, sli, guard, TK, TG, X0)
y = x - X0; pad = @(A) padarray(A, [TK TK], 'symmetric');
N = ring(pad(ones(size(x))), sli, guard, TK, TG); S1 = ring(pad(y), sli, guard, TK, TG); S2 = ring(pad(y.^2), sli, guard, TK, TG);
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
function B = blockany(A, f)
[h, w] = size(A); h = floor(h/f)*f; w = floor(w/f)*f; B = squeeze(any(any(reshape(A(1:h,1:w), f, h/f, f, w/f), 1), 3));
end
