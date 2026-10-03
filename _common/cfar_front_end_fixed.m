function fe = cfar_front_end_fixed(I, sli, guard, lut_dir, varargin)
%CFAR_FRONT_END_FIXED  Bit-accurate fixed-point model of the SHARED front end.
%
%   fe = CFAR_FRONT_END_FIXED(I, sli, guard, lut_dir)
%
%   The fixed-point counterpart of cfar_front_end.m, and the block that is
%   identical for all five detectors. Uses ONLY:
%     * one table lookup per pixel (log_amp_lut: I -> xc, centred)
%     * integer add / subtract / multiply
%     * reciprocal-multiply by compile-time constants (1/N, 1/(N-1), ...)
%   No log, sqrt, exp, division or floating-point arithmetic at "run time".
%
%   ---------------------------------------------------------------------
%   WHY THE BOX SUMS ARE SEPARABLE MOVING SUMS, NOT AN INTEGRAL IMAGE
%   ---------------------------------------------------------------------
%   cfar_front_end.m uses a summed-area table, which is the fastest way to do
%   this in MATLAB but is NOT what the hardware does and is not safe to model
%   in fixed point: an integral image accumulates over the whole image, so
%   its intermediate magnitudes grow with image size and would need a word
%   length that depends on the frame dimensions.
%
%   A separable moving sum -- a k-tap running sum along each row, then down k
%   rows -- bounds every intermediate by (window size) x (max term),
%   independent of image size. That is exactly the line-buffer /
%   shift-register accumulator the RTL implements, and it is why the
%   accumulator widths in fixedpoint_config.m can be stated as a function of
%   (sli, guard) alone.
%
%   ---------------------------------------------------------------------
%   THE THIRD MOMENT
%   ---------------------------------------------------------------------
%   This is the part the existing Weibull RTL does not have, and the part
%   that carries the Phase 4 DSP risk. c3 is formed from raw power sums as
%       m3 = S3/N - 3*m1*(S2/N) + 2*m1^3
%   which is only numerically viable because every sum is taken on the
%   CENTRED xc (BUG_LOG H1) -- without that, the three terms are each of
%   order 8 and cancel to a result of order 0.01, losing ~3 decimal digits
%   that Q-format arithmetic does not have to spare.
%
%   INPUTS
%     I       : HxW image, 8-bit-valued
%     sli     : window size (odd, >=3)     -- must match the LUT's geometry
%     guard   : guard size (odd, >=1, <sli)
%     lut_dir : folder holding log_amp_lut.mat (from generate_shared_luts)
%
%   NAME-VALUE OPTIONS
%     'Config' : a fixedpoint_config struct (default: built from sli/guard)
%
%   OUTPUT (struct fe) -- both the integer codes and the dequantised values,
%   so a caller can compare against the floating model directly:
%     x_code, c1_code, c2_code, c3_code   integer (the hardware signals)
%     x, c1, c2, c3, skew                  dequantised doubles
%     N, sli, guard, cfg
%     Overflow                             struct of per-stage overflow counts
%
%   See also: cfar_front_end, fixedpoint_config, generate_shared_luts

p = inputParser;
addParameter(p, 'Config', [], @(v) isempty(v) || isstruct(v));
parse(p, varargin{:});
cfg = p.Results.Config;
if isempty(cfg)
    cfg = fixedpoint_config(sli, guard);
end

if mod(sli,2)==0 || sli<3,  error('cfar_front_end_fixed:BadWindow','sli must be odd and >=3.'); end
if mod(guard,2)==0 || guard<1 || guard>=sli
    error('cfar_front_end_fixed:BadGuard','guard must be odd, >=1 and < sli.');
end

L = load(fullfile(lut_dir, 'log_amp_lut.mat'));
N = cfg.N;
I = double(I);
[h, w] = size(I);

ovf = struct('S1',0,'S2',0,'S3',0,'c2',0,'c3',0);

%% ---- Stage 1: log-amplitude by table lookup ----------------------------
% The ONLY place the raw pixel value is used. The table already has X0
% subtracted, so everything downstream works on the centred quantity.
idx    = min(max(round(I), 0), 255) + 1;
xc_code = L.code(idx);                      % Q1.14 signed integer

%% ---- Stage 2: powers, requantised before accumulation ------------------
% xc^2 : product is at scale 2^(2*xc.frac); bring it back to xc2.frac.
xc2_code = round(double(xc_code).^2 / 2^(2*cfg.xc.frac - cfg.xc2.frac));
% xc^3 : same idea, one more multiply.
xc3_code = round(double(xc_code).^3 / 2^(3*cfg.xc.frac - cfg.xc3.frac));

%% ---- Stage 3: reference-cell sums via separable moving sums ------------
S1 = box_sum_fixed(xc_code,  sli) - box_sum_fixed(xc_code,  guard);
S2 = box_sum_fixed(xc2_code, sli) - box_sum_fixed(xc2_code, guard);
S3 = box_sum_fixed(xc3_code, sli) - box_sum_fixed(xc3_code, guard);

ovf.S1 = count_ovf(S1, cfg.S1);
ovf.S2 = count_ovf(S2, cfg.S2);
ovf.S3 = count_ovf(S3, cfg.S3);

%% ---- Stage 4: moments, by reciprocal-multiply --------------------------
% m1 = S1/N. N is fixed once the geometry is chosen, so 1/N is one stored
% constant and this is a multiply-and-shift, not a divider.
M  = cfg.RECIP_M;

% m1 is carried at cfg.m1.frac (24), NOT at xc.frac (14). See the note on
% C.m1 in fixedpoint_config.m: m1 truncated to xc's width was the dominant
% error in BOTH c2 and c3, because m1^2 and m1^3 feed the cancellation
% directly. S1 is at xc.frac, so promoting costs one shift.
m1_code = round(S1 .* cfg.invN / 2^M * 2^(cfg.m1.frac - cfg.xc.frac));  % scale m1.frac

% m2 = S2/N - m1^2
s2n_code  = round(S2 .* cfg.invN / 2^M);                     % scale xc2.frac
m1sq_code = round(double(m1_code).^2 / 2^(2*cfg.m1.frac - cfg.xc2.frac));
m2_code   = s2n_code - m1sq_code;                            % scale xc2.frac

% m3 = S3/N - 3*m1*(S2/N) + 2*m1^3
s3n_code   = round(S3 .* cfg.invN / 2^M);                    % scale xc3.frac
m1s2_code  = round(double(m1_code) .* double(s2n_code) ...
                   / 2^(cfg.m1.frac + cfg.xc2.frac - cfg.xc3.frac));
m1cu_code  = round(double(m1_code).^3 / 2^(3*cfg.m1.frac - cfg.xc3.frac));
m3_code    = s3n_code - 3*m1s2_code + 2*m1cu_code;           % scale xc3.frac

%% ---- Stage 5: unbiased k-statistics ------------------------------------
% k2 = N/(N-1)*m2, k3 = N^2/((N-1)(N-2))*m3. Both scale factors are
% compile-time constants; stored to RECIP_M fractional bits and applied as a
% multiply-and-shift like everything else.
k2_mult = round(cfg.k2_scale * 2^M);
k3_mult = round(cfg.k3_scale * 2^M);

c2_code_raw = round(double(m2_code) .* k2_mult / 2^M);       % scale xc2.frac
c3_code_raw = round(double(m3_code) .* k3_mult / 2^M);       % scale xc3.frac

% Requantise c2 and c3 into their own declared formats.
c2_code = round(double(c2_code_raw) * 2^(cfg.c2.frac - cfg.xc2.frac));
c3_code = round(double(c3_code_raw) * 2^(cfg.c3.frac - cfg.xc3.frac));

% c2 must be non-negative: a negative value here is floating-point-style
% cancellation on a flat window, which in fixed point shows up as a small
% negative code. Floor at zero rather than letting a negative propagate into
% a sqrt or a LUT address.
c2_code = max(c2_code, 0);

ovf.c2 = count_ovf_code(c2_code, cfg.c2);
ovf.c3 = count_ovf_code(c3_code, cfg.c3);

%% ---- Stage 6: c1 back in the uncentred domain --------------------------
% c1 = m1 + X0. X0 is a constant, so this is one add. m1 now arrives at
% m1.frac, so X0 is scaled to match before the add.
X0_code = round(cfg.X0 * cfg.m1.scale);
c1_code_m1frac = double(m1_code) + X0_code;
c1_code = round(c1_code_m1frac * 2^(cfg.c1.frac - cfg.m1.frac));

%% ---- Dequantise for inspection / comparison ----------------------------
fe = struct();
fe.x_code  = xc_code;
fe.c1_code = c1_code;
fe.c2_code = c2_code;
fe.c3_code = c3_code;

fe.x    = double(xc_code) / cfg.xc.scale + cfg.X0;   % uncentred, matches float
fe.c1   = double(c1_code) / cfg.c1.scale;
fe.c2   = double(c2_code) / cfg.c2.scale;
fe.c3   = double(c3_code) / cfg.c3.scale;
fe.skew = fe.c3 ./ max(fe.c2, cfg.c2.res).^1.5;

fe.N = N; fe.sli = sli; fe.guard = guard;
fe.cfg = cfg;
fe.Overflow = ovf;
end


%% =======================================================================
function S = box_sum_fixed(v_code, k)
%BOX_SUM_FIXED  k x k box sum as a SEPARABLE moving sum (row pass, then
%   column pass), so every intermediate is bounded by k*max|term| and is
%   independent of image size -- matching the RTL line-buffer accumulator.
    tk = (k - 1) / 2;
    vp = padarray(double(v_code), [tk tk], 'symmetric', 'both');
    rp = movsum(vp, k, 2, 'Endpoints', 'discard');
    S  = movsum(rp, k, 1, 'Endpoints', 'discard');
end

function n = count_ovf(S, f)
%COUNT_OVF  How many accumulator values exceed the declared width.
    if f.signed
        lo = -2^(f.total-1); hi = 2^(f.total-1) - 1;
    else
        lo = 0; hi = 2^f.total - 1;
    end
    n = sum(S(:) < lo | S(:) > hi);
end

function n = count_ovf_code(c, f)
    n = count_ovf(c, f);
end
