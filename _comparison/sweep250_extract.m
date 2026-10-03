function sweep250_extract(varargin)
%SWEEP250_EXTRACT  Bit-exact (fixed-point) Weibull detection maps for a list of HRSID images, ALL FOUR Pfa planes.
%
%   Paper 2 cascade-vs-Weibull sweep, step 1 of 2 (step 2 = cnn/sweep250/sweep250.py).
%   Uses exactly the detection arithmetic of extract_cnn_patches_hwspec('Fixed',true) (== the RTL), but keeps the
%   DETECTION MAPS so Python can score Weibull-only and cascade output on the same pixel-level Pd/Pfa definition.
%
%   Per image and Pfa plane p (1..4 <-> 1e-3,1e-4,1e-5,1e-6) it saves the sparse detection map:
%       Didx{k,p}  uint32 column-major linear indices of detected pixels (the RTL's first-interior-pixel quirk applied)
%       Dg{k,p}    int32  hardware gate at those pixels, Q.14:  x_code + 19866 - 2*c1_code
%   plus h/w, the GT boxes [x1 y1 x2 y2] (1-based, same convention as extract_cnn_patches_hwspec) and the image name.
%
%   usage (from F:/Projects/CFAR):  matlab -batch "cd('F:/Projects/CFAR/_comparison'); sweep250_extract"
%   input : Results/sweep250/img_list.txt   (1-based HRSID directory indices, one per line; made by sweep250.py --make-list)
%   output: Results/sweep250/det_maps.mat

p = inputParser;
addParameter(p, 'Sli', 17); addParameter(p, 'Guard', 13);
addParameter(p, 'ListFile', 'sweep250/img_list.txt');
addParameter(p, 'OutName', 'sweep250/det_maps.mat');
parse(p, varargin{:});
paths = cfar_setup();
sli = p.Results.Sli; guard = p.Results.Guard; TK = (sli-1)/2;
lst = fullfile(paths.results, p.Results.ListFile);
imgList = round(load(lst));
imgs = dir(fullfile(paths.hrsidImages, '*.png'));
boxMap = load_coco_boxes(paths.hrsidAnnFile);

cfgFx = fixedpoint_config(sli, guard);
sharedDir = fullfile(paths.root, 'lut', 'shared');
wl = load(fullfile(paths.root, 'lut', 'weibull', 'weibull_delta_lut.mat'));
nlut = wl.entries_per_pfa;
nP = 4;

n = numel(imgList);
Didx = cell(n, nP); Dg = cell(n, nP); H = zeros(n,1); W = zeros(n,1);
names = cell(n,1); gts = cell(n,1);
t0 = tic;
for k = 1:n
    ii = imgList(k);
    name = imgs(ii).name; names{k} = name;
    if isKey(boxMap, name), gts{k} = boxMap(name); else, gts{k} = zeros(0,4); end
    I = double(imread(fullfile(paths.hrsidImages, name)));
    if ndims(I) == 3, I = I(:,:,1); end
    h = floor(size(I,1)/2)*2; w = floor(size(I,2)/2)*2; I = I(1:h,1:w);
    H(k) = h; W(k) = w;
    fe = cfar_front_end_fixed(I, sli, guard, sharedDir, 'Config', cfgFx);
    c2r = fe.c2(:);
    aidx = min(max(round((c2r - wl.addr_min) / (wl.addr_max - wl.addr_min) * (nlut-1)), 0), nlut-1);
    c1c = double(fe.c1_code(:)); c1t = sign(c1c) .* floor((abs(c1c) + 4) / 8);   % rshift_round(.,3)
    v = false(h, w); v(TK+1:end-TK, TK+1:end-TK) = true;
    gq14 = double(fe.x_code(:)) + 19866 - 2*double(fe.c1_code(:));
    for pi_ = 1:nP
        dn = double(wl.code((pi_-1)*nlut + aidx + 1));
        dc = sign(dn) .* floor((abs(dn) + 2) / 4);                               % rshift_round(.,2)
        Tc = (c1t + dc - 1242) * 16;
        D = reshape(double(fe.x_code(:)) > Tc, [h w]) & v;
        D(TK+1, TK+1) = false;                                                   % the RTL never emits the first interior pixel
        li = find(D);
        Didx{k,pi_} = uint32(li);
        Dg{k,pi_}   = int32(gq14(li));
    end
    if mod(k, 10) == 0, fprintf('  [%3d/%3d] %.0fs\n', k, n, toc(t0)); end
end
meta = struct('Sli', sli, 'Guard', guard, 'PfaPlanes', [1e-3 1e-4 1e-5 1e-6], 'Fixed', true, 'imgList', imgList);
outf = fullfile(paths.results, p.Results.OutName);
save(outf, 'Didx', 'Dg', 'H', 'W', 'names', 'gts', 'meta', '-v7.3');
fprintf('wrote %s (%.0fs)\n', outf, toc(t0));
end
