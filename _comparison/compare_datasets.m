function compare_datasets(varargin)
%COMPARE_DATASETS  Is the 3-parameter estimators' failure rate a property of
%   SAR clutter, or of the SSDD dataset's 8-bit encoding?
%
%   COMPARE_DATASETS()
%
%   Runs the same front end and the same support tests over BOTH datasets:
%     SSDD  -- 8-bit JPEG-encoded SAR screenshots with ground truth
%     MSTAR -- genuine complex-valued SAR chips (Phoenix format), magnitude
%              rescaled to the same 0..255 intensity range
%
%   The two differ in physics not at all -- both are X-band SAR amplitude --
%   but differ completely in encoding. If the Burr-XII and Generalized-Gamma
%   support conditions fail on one and not the other, the failure is an
%   artefact of the encoding, and the conclusion for Phase 3/4 is that the
%   3-parameter detectors should be judged on MSTAR-like data, not written off
%   on the basis of SSDD alone.
%
%   Writes Results/dataset_comparison.csv

p = inputParser;
addParameter(p, 'Sli',   21);
addParameter(p, 'Guard', 15);
addParameter(p, 'NumSSDD', 40);
parse(p, varargin{:});
sli = p.Results.Sli; guard = p.Results.Guard;

paths = cfar_setup();
BURR_LO = -1.139443;

fprintf('=====================================================================\n');
fprintf(' Clutter log-skewness: SSDD (8-bit JPEG) vs MSTAR (native complex SAR)\n');
fprintf('   sli=%d guard=%d\n', sli, guard);
fprintf('=====================================================================\n');

rows = struct('Dataset',{},'Source',{},'MedianC2',{},'MedianSkew',{}, ...
              'PctBelowBurr',{},'PctOutsideGGexact',{},'PctDarkPixels',{});

%% ---- SSDD ---------------------------------------------------------------
imgs = dir(fullfile(paths.ssddJpeg,'*.jpg'));
idx = unique(round(linspace(1, numel(imgs), min(p.Results.NumSSDD, numel(imgs)))));
SKa = []; C2a = []; DKa = [];
for k = idx
    Ik = imread(fullfile(paths.ssddJpeg, imgs(k).name));
    if ndims(Ik)==3, Ik = rgb2gray(Ik); end
    Id = double(Ik);
    fe = cfar_front_end(Id, sli, guard);
    SKa=[SKa; fe.skew(1:9:end)']; C2a=[C2a; fe.c2(1:9:end)']; %#ok<AGROW>
    DKa=[DKa; mean(Id(:)<=2)];                                 %#ok<AGROW>
end
rows(end+1) = mkrow('SSDD', sprintf('%d images', numel(idx)), C2a, SKa, mean(DKa), BURR_LO);

%% ---- MSTAR --------------------------------------------------------------
chips = dir(fullfile(paths.mstar, 'HB*'));
SKb = []; C2b = []; DKb = [];
for k = 1:numel(chips)
    st = MSTAR_LOAD_IMAGE(fullfile(paths.mstar, chips(k).name));
    A = abs(st.ImageData);
    % Same rescaling the MSTAR demo drivers use, so the front end sees the
    % same kind of 0..255 intensity in both cases -- the comparison is then
    % about the ENCODING (8-bit JPEG vs native float), not about units.
    Id = 255 * (A - min(A(:))) / max(max(A(:)) - min(A(:)), eps);
    fe = cfar_front_end(Id, sli, guard);
    SKb=[SKb; fe.skew(1:9:end)']; C2b=[C2b; fe.c2(1:9:end)']; %#ok<AGROW>
    DKb=[DKb; mean(Id(:)<=2)];                                 %#ok<AGROW>
end
rows(end+1) = mkrow('MSTAR', sprintf('%d chips', numel(chips)), C2b, SKb, mean(DKb), BURR_LO);

%% ---- MSTAR, requantised to 8 bits --------------------------------------
% The decisive control: take the SAME MSTAR data and round it to integers,
% which is the one thing SSDD does that MSTAR does not. If the skewness then
% collapses toward SSDD's, the cause is the quantization and nothing else.
SKc = []; C2c = []; DKc = [];
for k = 1:numel(chips)
    st = MSTAR_LOAD_IMAGE(fullfile(paths.mstar, chips(k).name));
    A = abs(st.ImageData);
    Id = round(255 * (A - min(A(:))) / max(max(A(:)) - min(A(:)), eps));
    fe = cfar_front_end(Id, sli, guard);
    SKc=[SKc; fe.skew(1:9:end)']; C2c=[C2c; fe.c2(1:9:end)']; %#ok<AGROW>
    DKc=[DKc; mean(Id(:)<=2)];                                 %#ok<AGROW>
end
rows(end+1) = mkrow('MSTAR-8bit', sprintf('%d chips, rounded', numel(chips)), ...
                    C2c, SKc, mean(DKc), BURR_LO);

%% ---- SARFish (Sentinel-1 GRD, genuine spaceborne, native 16-bit) --------
% A second, independent real-SAR data point alongside MSTAR (airborne,
% X-band, Phoenix format). MSTAR answers "is the SSDD failure a JPEG
% artefact"; SARFish answers "...and does that hold for a completely
% different platform/band/processor too, or is MSTAR itself special".
% No ship ground truth needed here -- this test is about background CLUTTER
% statistics (c2, skew), which JPEG corrupts whether or not a target is in
% the crop. See sarfish_sample/extract_crops.py for how these were pulled.
if isfolder(paths.sarfish)
    crops = dir(fullfile(paths.sarfish, '*.mat'));
    if isempty(crops)
        warning('compare_datasets:NoSARFish', ...
            'sarfish_sample/crops has no .mat files -- run extract_crops.py first. Skipping SARFish arm.');
    else
        SKd = []; C2d = []; DKd = [];
        for k = 1:numel(crops)
            S = load(fullfile(crops(k).folder, crops(k).name), 'img');
            A = S.img;  % native uint16 range, already double
            % Same 0..255 rescaling MSTAR gets, so all three real-data arms
            % are compared in the same units.
            Id = 255 * (A - min(A(:))) / max(max(A(:)) - min(A(:)), eps);
            fe = cfar_front_end(Id, sli, guard);
            SKd=[SKd; fe.skew(1:9:end)']; C2d=[C2d; fe.c2(1:9:end)']; %#ok<AGROW>
            DKd=[DKd; mean(Id(:)<=2)];                                 %#ok<AGROW>
        end
        rows(end+1) = mkrow('SARFish', sprintf('%d crops, S1 GRD native', numel(crops)), ...
                            C2d, SKd, mean(DKd), BURR_LO);
    end
else
    warning('compare_datasets:NoSARFish', ...
        'sarfish_sample/crops not found -- run sarfish_sample/download_sample.py and extract_crops.py first. Skipping SARFish arm.');
end

T = struct2table(rows);
fprintf('\n%-14s %-18s %10s %11s %14s %17s %13s\n', 'Dataset','Source','med c2', ...
    'med skew','%% below Burr','%% outside GG-exact','%% dark px');
for i = 1:height(T)
    fprintf('%-14s %-18s %10.4f %11.3f %13.1f%% %16.1f%% %12.1f%%\n', ...
        T.Dataset{i}, T.Source{i}, T.MedianC2(i), T.MedianSkew(i), ...
        T.PctBelowBurr(i), T.PctOutsideGGexact(i), 100*T.PctDarkPixels(i));
end
fprintf('\nPure Weibull clutter has skew = %.6f exactly (Burr''s lower support limit).\n', BURR_LO);

writetable(T, fullfile(paths.results,'dataset_comparison.csv'));
fprintf('Written: %s\n', fullfile(paths.results,'dataset_comparison.csv'));
end

function r = mkrow(ds, src, c2, sk, darkFrac, burrLo)
    c2 = c2(:); sk = sk(:);
    ok = isfinite(sk) & isfinite(c2);
    r = struct('Dataset', ds, 'Source', src, ...
        'MedianC2', median(c2(ok)), 'MedianSkew', median(sk(ok)), ...
        'PctBelowBurr', 100*mean(sk(ok) < burrLo), ...
        'PctOutsideGGexact', 100*mean(sk(ok).^2 > 4), ...
        'PctDarkPixels', darkFrac);
end
