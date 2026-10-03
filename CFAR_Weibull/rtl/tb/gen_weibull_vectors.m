function gen_weibull_vectors()
paths = cfar_setup();
sli = 17; guard = 13;
cfg = fixedpoint_config(sli, guard);
shared_dir = fullfile(paths.root,'lut','shared');
ln_dir     = fullfile(paths.root,'lut','weibull');
lut = load(fullfile(ln_dir, 'weibull_delta_lut.mat'));

imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
I = double(imread(fullfile(paths.ssddJpeg, imgs(12).name)));
if ndims(I)==3, I = rgb2gray(uint8(I)); I = double(I); end
I = I(1:60, 1:60);
fe = cfar_front_end_fixed(I, sli, guard, shared_dir, 'Config', cfg);

% *** cfar_fixed_decide.m addresses on the DEQUANTISED REAL c2 (fe.c2),
% NOT the raw integer code (fe.c2_code) -- LognormalCFAR_FixedPoint.m calls
% cfar_fixed_decide(fe, lut, fe.c2, ...) explicitly. Addressing on the raw
% code here (an earlier version of this script) multiplies by ~2^24 too
% much, sending the address into the billions before the clamp -- confirmed
% by hand-checking one mismatching case directly in MATLAB, which is what
% caught this before spending more time doubting the RTL. c2_code (the raw
% integer) is still what feeds the RTL, unchanged; only the ADDRESS
% computation uses the real value, matching cfar_fixed_decide.m.
c2_code = fe.c2_code(:);
c2_real = fe.c2(:);
pfaIdx = mod((0:numel(c2_code)-1)', lut.n_pfa) + 1;

n = lut.entries_per_pfa; vmin = lut.addr_min; vmax = lut.addr_max;
a = round((c2_real - vmin) / (vmax - vmin) * (n-1));
a = min(max(a, 0), n-1);
flat = (pfaIdx-1)*n + a + 1;
delta_code = lut.code(flat);

outdir = fullfile(paths.root, 'CFAR_Weibull', 'rtl', 'tb');
fid = fopen(fullfile(outdir, 'weibull_vectors.txt'), 'w');
fprintf(fid, '%% c2_code pfa_sel(0idx) delta_code\n');
for i = 1:numel(c2_code)
    fprintf(fid, '%d %d %d\n', c2_code(i), pfaIdx(i)-1, delta_code(i));
end
fclose(fid);
fprintf('Wrote %d vectors.\n', numel(c2_code));
end
