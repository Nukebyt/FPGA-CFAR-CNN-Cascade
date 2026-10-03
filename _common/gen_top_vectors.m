function gen_top_vectors(detector)
%GEN_TOP_VECTORS  Full-chain (pixel-in -> detect/T_log) golden vectors for
%   each detector's <name>_top.v integration testbench.
%
%   gen_top_vectors('gengamma' | 'burr' | 'lognormal' | 'weibull' | 'g0')
%
%   Reuses the SAME image/crop (SSDD image #37, 40x48, sli=17/guard=13) as
%   gen_frontend3_vectors.m, whose (c1,c2,c3) output is already verified
%   bit-exact against front_end3.v (767/767, front_end3_tb.v) -- so this
%   script only needs to add each detector's OWN delta formula (copied from
%   that detector's existing gen_<name>_vectors.m, which already verified the
%   backend in isolation) plus a bit-exact MATLAB replica of
%   threshold_compare3.v's integer arithmetic (rshift_round, the X0 recentring
%   subtraction, the final shifted compare) -- so a mismatch here means the
%   INTEGRATION wiring (delay depths, delta rescale-to-Q5.10) is wrong, not
%   that any already-verified piece is wrong.
%
%   pfa_sel is held CONSTANT (=1) for the whole run, matching how the real
%   top-level module actually uses it (a single external input, not part of
%   the per-pixel pipeline) -- unlike each backend's own isolated test, which
%   cycles pfa_sel through all 4 values per sample purely to exercise every
%   ROM plane in one run. ROM-plane addressing was already exhaustively
%   verified there; this test's job is the glue between modules.

paths = cfar_setup();
sli = 17; guard = 13; TK = (sli-1)/2;
cfg = fixedpoint_config(sli, guard);
shared_dir = fullfile(paths.root,'lut','shared');

IMG_H = 40; IMG_W = 48;
imgs = dir(fullfile(paths.ssddJpeg, '*.jpg'));
Ifull = imread(fullfile(paths.ssddJpeg, imgs(37).name));
if ndims(Ifull) == 3, Ifull = rgb2gray(Ifull); end
I = double(Ifull(1:IMG_H, 1:IMG_W));

fe = cfar_front_end_fixed(I, sli, guard, shared_dir, 'Config', cfg);

PFA_SEL = 1;               % 0-indexed, held constant
pfaIdx  = PFA_SEL + 1;     % 1-indexed, for ROM lookups below

c2 = fe.c2_code(:); c3 = fe.c3_code(:); c2r = fe.c2(:);

% ---- per-detector delta, native code + native frac bits -------------------
switch detector
case 'gengamma'
    d = fullfile(paths.root,'lut','gengamma');
    Dlut = load(fullfile(d,'gengamma_num_lut.mat'));
    ispLut = load(fullfile(d,'gengamma_invsqrtpsi_lut.mat'));
    meta = load(fullfile(d,'gengamma_lut_meta.mat'));
    sqMant = load(fullfile(shared_dir,'sqrt_mant_lut.mat'));
    mantLut = load(fullfile(shared_dir,'log2_mant_lut.mat'));

    lg_c2 = log2_fixed(c2, cfg.c2.frac, mantLut);
    lg_c3 = log2_fixed(abs(c3), cfg.c3.frac, mantLut);
    lg_s = lg_c3 - 1.5*lg_c2;
    sgn = sign(-double(c3));
    n = Dlut.entries_per_pfa; nP = Dlut.n_pfa;
    lo = Dlut.addr_min; hi = Dlut.addr_max;
    a = round((lg_s - lo) / (hi - lo) * (n-1));
    a = min(max(a, 0), n-1); a(~isfinite(a)) = 0;
    sq = sqrt_fixed(c2, cfg.c2.frac, sqMant);
    isp = double(ispLut.code(a+1)) / ispLut.format.scale;
    invAbsV = sq .* isp;
    invAbsV = min(max(invAbsV, 1/meta.VMax), 1/meta.VMin);
    sign_sel = double(sgn < 0);
    flat = ((sign_sel*nP) + (pfaIdx-1))*n + a + 1;
    Num = double(Dlut.code(flat)) / Dlut.format.scale;
    delta = Num .* invAbsV;
    delta_native = round(delta * cfg.delta.scale);
    delta_native = max(min(delta_native, cfg.delta.max*cfg.delta.scale), cfg.delta.min*cfg.delta.scale);
    NATIVE_FRAC = cfg.delta.frac; % 10

case 'burr'
    d = fullfile(paths.root,'lut','burr');
    Nlut = load(fullfile(d,'burr_num_lut.mat'));
    Ilut = load(fullfile(d,'burr_invsqrtpsi_lut.mat'));
    sqMant = load(fullfile(shared_dir,'sqrt_mant_lut.mat'));
    mantLut = load(fullfile(shared_dir,'log2_mant_lut.mat'));

    lg_c2 = log2_fixed(c2, cfg.c2.frac, mantLut);
    lg_c3 = log2_fixed(abs(c3), cfg.c3.frac, mantLut);
    lg_s = lg_c3 - 1.5*lg_c2;
    sign_sel = double(c3 < 0);
    n = Nlut.entries_per_pfa; nP = Nlut.n_pfa;
    lo = Nlut.addr_min; hi = Nlut.addr_max;
    a = round((lg_s - lo) / (hi - lo) * (n-1));
    a = min(max(a, 0), n-1); a(~isfinite(a)) = 0;
    sq = sqrt_fixed(c2, cfg.c2.frac, sqMant);
    isp = double(Ilut.code(sign_sel*n + a + 1)) / Ilut.format.scale;
    invAbsRho = sq .* isp;
    invAbsRho = min(max(invAbsRho, 1/50), 1/0.05);
    flat = ((sign_sel*nP) + (pfaIdx-1))*n + a + 1;
    Num = double(Nlut.code(flat)) / Nlut.format.scale;
    delta = Num .* invAbsRho;
    delta_native = round(delta * cfg.delta.scale);
    delta_native = max(min(delta_native, cfg.delta.max*cfg.delta.scale), cfg.delta.min*cfg.delta.scale);
    NATIVE_FRAC = cfg.delta.frac; % 10

case 'lognormal'
    d = fullfile(paths.root,'lut','lognormal');
    lut = load(fullfile(d,'lognormal_delta_lut.mat'));
    n = lut.entries_per_pfa; vmin = lut.addr_min; vmax = lut.addr_max;
    a = round((c2r - vmin) / (vmax - vmin) * (n-1));
    a = min(max(a, 0), n-1);
    flat = (pfaIdx-1)*n + a + 1;
    delta_native = double(lut.code(flat));
    NATIVE_FRAC = 11; % Q4.11, per lognormal_delta_lut.txt

case 'weibull'
    d = fullfile(paths.root,'lut','weibull');
    lut = load(fullfile(d,'weibull_delta_lut.mat'));
    n = lut.entries_per_pfa; vmin = lut.addr_min; vmax = lut.addr_max;
    a = round((c2r - vmin) / (vmax - vmin) * (n-1));
    a = min(max(a, 0), n-1);
    flat = (pfaIdx-1)*n + a + 1;
    delta_native = double(lut.code(flat));
    NATIVE_FRAC = 12; % Q3.12, per weibull_delta_lut.txt

case 'g0'
    d = fullfile(paths.root,'lut','g0');
    Dlut = load(fullfile(d,'g0_delta_lut.mat'));
    Glut = load(fullfile(d,'g0_lgdmax_lut.mat'));
    mantLut = load(fullfile(shared_dir,'log2_mant_lut.mat'));

    lg_c2 = log2_fixed(c2, cfg.c2.frac, mantLut);
    lg_c3 = log2_fixed(abs(c3), cfg.c3.frac, mantLut);
    N1 = Dlut.N1; N2 = Dlut.N2;
    a1_lo = Dlut.addr1_min; a1_hi = Dlut.addr1_max;
    a2_lo = Dlut.addr2_min; a2_hi = Dlut.addr2_max;
    f1 = (lg_c2 - a1_lo) / (a1_hi - a1_lo) * (N1 - 1);
    f1 = min(max(f1, 0), N1 - 1 - 1e-9);
    i1 = floor(f1); w1 = f1 - i1;
    lgd0 = double(Glut.code(i1 + 1)) / Glut.format.scale;
    lgd1 = double(Glut.code(min(i1 + 1, N1 - 1) + 1)) / Glut.format.scale;
    lgDmax = lgd0 .* (1 - w1) + lgd1 .* w1;
    lg_dnorm = (3 + lg_c3) - lgDmax;
    f2 = (lg_dnorm - a2_lo) / (a2_hi - a2_lo) * (N2 - 1);
    f2 = min(max(f2, 0), N2 - 1 - 1e-9);
    f2(~isfinite(f2)) = 0;
    i2 = floor(f2); w2 = f2 - i2;
    sign_sel = double(c3 < 0);
    base = ((pfaIdx-1)*2 + sign_sel) * N2;
    i1b = min(i1 + 1, N1 - 1); i2b = min(i2 + 1, N2 - 1);
    d00 = double(Dlut.code((base + i2 )*N1 + i1  + 1)) / Dlut.format.scale;
    d10 = double(Dlut.code((base + i2 )*N1 + i1b + 1)) / Dlut.format.scale;
    d01 = double(Dlut.code((base + i2b)*N1 + i1  + 1)) / Dlut.format.scale;
    d11 = double(Dlut.code((base + i2b)*N1 + i1b + 1)) / Dlut.format.scale;
    delta = (d00.*(1-w1) + d10.*w1).*(1-w2) + (d01.*(1-w1) + d11.*w1).*w2;
    delta_native = round(delta * Dlut.format.scale);
    NATIVE_FRAC = 18; % Q5.18, per g0_delta_lut.txt

otherwise
    error('gen_top_vectors:bad_detector', 'unknown detector %s', detector);
end

% ---- rescale to Q5.10, bit-exact match to each <name>_top.v's rshift_round
TARGET_FRAC = 10;
sh = NATIVE_FRAC - TARGET_FRAC;
delta_code = rshift_round_ref(delta_native, sh);

% ---- replicate threshold_compare3.v's integer arithmetic exactly ----------
C1_FRAC = 13; TLOG_FRAC = 10; X_FRAC = 14;
SH_C1 = C1_FRAC - TLOG_FRAC; % 3
X0_CODE_TLOGFRAC = 1242;
SH_CMP = X_FRAC - TLOG_FRAC; % 4

c1 = fe.c1_code(:);
c1_at_tlog = rshift_round_ref(c1, SH_C1);
T_log_uncentred = c1_at_tlog + delta_code;
T_log_centred = T_log_uncentred - X0_CODE_TLOGFRAC;
T_log_cmp = T_log_centred * (2^SH_CMP);
x = fe.x_code(:);
detect = x > T_log_cmp;

% ---- write pixel stream (shared with front_end3, but self-contained here) -
outdir_map = struct('gengamma', fullfile(paths.root,'CFAR Generalized Gamma','rtl','tb'), ...
                     'burr',     fullfile(paths.root,'CFAR_Burr','rtl','tb'), ...
                     'lognormal',fullfile(paths.root,'CFAR Lognormal','rtl','tb'), ...
                     'weibull',  fullfile(paths.root,'CFAR_Weibull','rtl','tb'), ...
                     'g0',       fullfile(paths.root,'CFAR_G0','rtl','tb'));
outdir = outdir_map.(detector);
if ~isfolder(outdir), mkdir(outdir); end

fid = fopen(fullfile(outdir, 'top_pixels.txt'), 'w');
for r = 1:IMG_H
    for c = 1:IMG_W
        fprintf(fid, '%d\n', I(r,c));
    end
end
fclose(fid);

fid = fopen(fullfile(outdir, 'top_expected.txt'), 'w');
fprintf(fid, '%% row col detect T_log_code\n');
for r = TK+1 : IMG_H-TK
    for c = TK+1 : IMG_W-TK
        % fe.*_code(:) linearises COLUMN-MAJOR (MATLAB default), so the flat
        % index for (row r, col c) is (c-1)*IMG_H + r.
        idx = (c-1)*IMG_H + r;
        fprintf(fid, '%d %d %d %d\n', r-1, c-1, detect(idx), T_log_uncentred(idx));
    end
end
fclose(fid);

fprintf('[%s] PFA_SEL=%d NATIVE_FRAC=%d SH=%d. Wrote top_pixels.txt (%d) and top_expected.txt (%d).\n', ...
    detector, PFA_SEL, NATIVE_FRAC, sh, IMG_H*IMG_W, (IMG_H-2*TK)*(IMG_W-2*TK));
fprintf('  detect fraction (interior) = %.4f\n', mean(detect( (mod(0:numel(detect)-1,IMG_H)+1>TK) )));
end

function r = rshift_round_ref(v, sh)
%RSHIFT_ROUND_REF  Bit-exact MATLAB mirror of every backend's Verilog
%   rshift_round function: arithmetic right shift by sh with round-half-up,
%   symmetric around zero (negate, shift, negate -- not a floor of the
%   signed value, which would round differently for negative inputs).
if sh <= 0
    r = v * (2^(-sh));
    return;
end
rb = 2^(sh-1);
r = zeros(size(v));
pos = v >= 0;
r(pos)  =  floor((v(pos)  + rb) / 2^sh);
r(~pos) = -floor((-v(~pos) + rb) / 2^sh);
end
