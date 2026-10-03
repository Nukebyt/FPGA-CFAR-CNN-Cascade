function count_cfar_clusters(varargin)
%COUNT_CFAR_CLUSTERS  Per-image count of ALL Weibull-prescreen clusters (no gate,
%   no patch extraction) -- the true CFAR-alone candidate count, needed because
%   extract_cnn_patches_hw.m with GateTau drops sub-gate negatives from its file.
%   Same image order as extract_cnn_patches_hw.m (sorted dir order).
p = inputParser;
addParameter(p, 'Sli', 17); addParameter(p, 'Guard', 13); addParameter(p, 'Pfa', 1e-3);
addParameter(p, 'OutName', 'cfar_cluster_counts.mat');
parse(p, varargin{:});
paths = cfar_setup();
imgs = dir(fullfile(paths.hrsidImages, '*.png'));
n = numel(imgs);
cnt = zeros(n, 1, 'uint32'); names = cell(n, 1);
t0 = tic;
for ii = 1:n
    names{ii} = imgs(ii).name;
    I = double(imread(fullfile(paths.hrsidImages, imgs(ii).name)));
    if ndims(I) == 3, I = I(:,:,1); end
    fe = cfar_front_end(I, p.Results.Sli, p.Results.Guard);
    prm = WeibullCFAR_Params(fe.c2);
    det = (fe.x > fe.c1 + WeibullCFAR_TLog(prm, p.Results.Pfa)) & prm.valid;
    cnt(ii) = bwconncomp(det, 8).NumObjects;
    if mod(ii, 500) == 0, fprintf('  [%d/%d] %.0fs\n', ii, n, toc(t0)); end
end
save(fullfile(paths.results, p.Results.OutName), 'cnt', 'names');
fprintf('Wrote %s  (mean %.1f clusters/img)\n', p.Results.OutName, mean(double(cnt)));
end
