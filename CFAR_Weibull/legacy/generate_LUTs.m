%% generate_LUTs.m
%
% Steps 3-5 of the Weibull CFAR FPGA prep pipeline: builds the three ROM
% look-up tables the streaming single-pass detector needs so it never has
% to compute log(), sqrt(), exp(), a fractional power, or a division at
% run time:
%
%   1) log_amp_lut       : LUT(I) = log(sqrt(I + 0.5)), I = 0..255   (Step 3)
%   2) c_lut              : quantized c2 -> Weibull shape parameter C (Step 4)
%   3) pfa_constant_lut   : Pfa selector -> K(Pfa) = gamma+log(-log(Pfa)) (Step 5)
%
% Each LUT is written as:
%   <name>.mat  - MATLAB arrays (for use elsewhere in this repo / re-plotting)
%   <name>.txt  - human-readable table, one row per entry
%   <name>.hex  - $readmemh / Quartus "Hex File" compatible: one value per
%                 line, MSB left, two's-complement encoding for signed
%                 formats. (If your Quartus ROM megafunction specifically
%                 wants classic record-based Intel HEX instead of this
%                 plain per-line format, that's a one-line change in
%                 write_hex_lut below -- flag it and it can be added.)
%
% This script deliberately calls WeibullCFAR_Floating.m to get its c1/c2
% maps (via stats.C2Map) rather than re-deriving the windowed statistics
% here, so the empirical analysis below always matches whatever
% WeibullCFAR_Floating.m currently does.
%
% Do NOT run this before WeibullCFAR_Floating.m exists on the path.

clc; clear all; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
repo_dir = fileparts(script_dir);          % .../WEIBULL_CFAR
lut_dir  = fullfile(repo_dir, 'lut');
if ~isfolder(lut_dir)
    mkdir(lut_dir);
end

C_MIN = 0.8;
C_MAX = 8.0;
PSI11 = pi^2 / 6;                 % trigamma(1), exact closed form
EULER_GAMMA = 0.5772156649015329; % Euler-Mascheroni constant

%% =========================================================================
%% STEP 3: LOG-AMPLITUDE LUT  —  LUT(I) = log(sqrt(I + 0.5)), I = 0..255
%% =========================================================================
fprintf('=== Step 3: Log-amplitude LUT ===\n');

I_values = (0:255)';
log_amp_lut_double = log(sqrt(I_values + 0.5));

lut_min = min(log_amp_lut_double);   % occurs at I=0 (function is monotonic)
lut_max = max(log_amp_lut_double);   % occurs at I=255
fprintf('LUT(I) = log(sqrt(I+0.5)) is monotonic increasing over I = 0..255.\n');
fprintf('  min = %.10f  (at I=0)\n',   lut_min);
fprintf('  max = %.10f  (at I=255)\n', lut_max);

% ---- Fixed-point format ----------------------------------------------
% Required range is [-0.3466, 2.7716] (computed above, not assumed) -> at
% least 2 integer bits are needed (2^2=4 > 2.7716) plus a sign bit. We use
% a 16-bit signed Q2.13 format:
%   1 sign bit + 2 integer bits + 13 fractional bits
%   representable range : [-4, 4 - 2^-13] = [-4, 3.999878...]
%   resolution           : 2^-13 = 1.2207e-4
% This leaves >4x headroom above the actual max and >10x margin below the
% min (in magnitude), with 13 fractional bits of precision -- far finer
% than needed given this value only ever feeds an average over N~100-400
% reference cells downstream.
LOGAMP_INT_BITS   = 2;
LOGAMP_FRAC_BITS  = 13;
LOGAMP_TOTAL_BITS = 1 + LOGAMP_INT_BITS + LOGAMP_FRAC_BITS;   % = 16, signed
LOGAMP_SCALE      = 2^LOGAMP_FRAC_BITS;

log_amp_lut_fixed_int = round(log_amp_lut_double * LOGAMP_SCALE);

max_code = 2^(LOGAMP_TOTAL_BITS-1) - 1;
min_code = -2^(LOGAMP_TOTAL_BITS-1);
assert(all(log_amp_lut_fixed_int <= max_code) && all(log_amp_lut_fixed_int >= min_code), ...
    'log_amp_lut: fixed-point overflow -- widen LOGAMP_INT_BITS.');

log_amp_lut_requantized = log_amp_lut_fixed_int / LOGAMP_SCALE;
log_amp_lut_error = log_amp_lut_requantized - log_amp_lut_double;
fprintf('  Format: signed Q%d.%d (%d-bit). Max quantization error: %.6e (resolution %.6e)\n', ...
    LOGAMP_INT_BITS, LOGAMP_FRAC_BITS, LOGAMP_TOTAL_BITS, ...
    max(abs(log_amp_lut_error)), 1/LOGAMP_SCALE);

save(fullfile(lut_dir, 'log_amp_lut.mat'), 'I_values', 'log_amp_lut_double', ...
    'log_amp_lut_fixed_int', 'LOGAMP_INT_BITS', 'LOGAMP_FRAC_BITS', 'LOGAMP_SCALE');

fid = fopen(fullfile(lut_dir, 'log_amp_lut.txt'), 'w');
fprintf(fid, '%% log_amp_lut : LUT(I) = log(sqrt(I+0.5)), I = 0..255\n');
fprintf(fid, '%% Format: signed Q%d.%d (%d-bit), scale = 2^%d = %d\n', ...
    LOGAMP_INT_BITS, LOGAMP_FRAC_BITS, LOGAMP_TOTAL_BITS, LOGAMP_FRAC_BITS, LOGAMP_SCALE);
fprintf(fid, '%% Columns: I  double_value  fixed_point_code(dec)  requantized_value\n');
for k = 1:numel(I_values)
    fprintf(fid, '%3d  %+.10f  %8d  %+.10f\n', I_values(k), log_amp_lut_double(k), ...
        log_amp_lut_fixed_int(k), log_amp_lut_requantized(k));
end
fclose(fid);

write_hex_lut(fullfile(lut_dir, 'log_amp_lut.hex'), log_amp_lut_fixed_int, LOGAMP_TOTAL_BITS, true);
fprintf('  Wrote log_amp_lut.mat / .txt / .hex to %s\n\n', lut_dir);

%% =========================================================================
%% STEP 4: WEIBULL SHAPE-PARAMETER (C) LUT  —  c2 -> C
%% =========================================================================
fprintf('=== Step 4: Weibull shape-parameter (C) LUT ===\n');

% ---- 4a. Analytical "core" c2 range ------------------------------------
% C = sqrt(PSI11/c2) is a strictly decreasing function of c2. The detector
% clamps C to [C_MIN, C_MAX] regardless of what the LUT says outside that
% range, so the only c2 range where the LUT output must actually VARY is
% where the unclamped C would itself land inside [C_MIN, C_MAX]. This
% range is exact, closed-form, and independent of the dataset:
c2_lo_analytical = PSI11 / C_MAX^2;   % c2 at which C_raw == C_MAX (=8.0)
c2_hi_analytical = PSI11 / C_MIN^2;   % c2 at which C_raw == C_MIN (=0.8)
fprintf('Analytical core c2 range (clamp not yet saturating): [%.6f, %.6f]\n', ...
    c2_lo_analytical, c2_hi_analytical);

% ---- 4b. Empirical c2 distribution over the SSDD dataset ---------------
% The analytical range above is exact but says nothing about which part
% of it real SAR clutter actually occupies, or whether outlier windows
% (e.g. still touching a bright target edge) push outside it. We sample
% the dataset with the ACTUAL WeibullCFAR_Floating.m model to check.
%
% IMPORTANT: the c2 distribution depends on sli/guard (more reference
% cells -> lower-variance c2 estimator -> tighter distribution). Keep
% sli_lut/guard_lut/Pfa_lut below in sync with whatever evaluate_dataset.m
% (and eventually the RTL) actually use, or regenerate this LUT if you
% change the window geometry.
dataset_dir = fullfile(repo_dir, 'BBox_SSDD', 'voc_style');
jpeg_dir    = fullfile(dataset_dir, 'JPEGImages');

% sli=51/guard=41 (0.80 ratio) replaces sli=21/guard=7 -- see the note at
% the top of evaluate_dataset.m and Results/param_sweep_results*.csv: the
% old guard was far smaller than typical SSDD ship extent (median ~37px),
% so target pixels leaked into the clutter reference cells. This geometry
% took pooled ship-level Pd from ~32% to ~89% on the full dataset.
%
% NOTE this is deliberately NOT the real hardware geometry (cfar_top.v runs
% SLI=18/GUARD=11) -- see README.md section 3.6 for why SLI is capped well
% below this software-optimal point. A same-session attempt to make this
% scan match the real hardware geometry exactly (SLI=17/GUARD=13 or 15,
% since SLI=18 is even and WeibullCFAR_Floating requires odd sli) is
% documented in full in rtl/BUG_REPORT.md #29/#30/#32: the resulting guard
% retune measurably improves Pd (63.3% -> 70-80% depending on guard), but
% every SLI=17 configuration tried hit an unresolvable Quartus Prime Lite
% 21.1 optimizer crash under default optimization, and every known
% workaround either broke timing closure project-wide or risked an ALM
% overflow -- confirmed empirically across 7 separate build attempts, not
% assumed. Reverted to the original, already-hardware-validated SLI=18/
% GUARD=11 for now. The accuracy improvement remains real and worth
% revisiting (candidates for a future attempt: SLI=18 with a GUARD other
% than 11, which was never tried this session and is a genuinely different
% combination from anything above).
sli_lut   = 51;
guard_lut = 41;
Pfa_lut   = 1e-3;

c2_samples = [];
if isfolder(jpeg_dir)
    imgs = dir(fullfile(jpeg_dir, '*.jpg'));
    if isempty(imgs)
        imgs = dir(fullfile(jpeg_dir, '*.jpeg'));
    end

    if isempty(imgs)
        warning('No JPEG images found in %s -- using the analytical range only.', jpeg_dir);
    else
        % Subsample images (and, within each, pixels) so this stays fast
        % while remaining representative; increase MAX_IMAGES / decrease
        % PIXEL_STRIDE for a more exhaustive scan.
        MAX_IMAGES   = min(60, numel(imgs));
        PIXEL_STRIDE = 4;   % keep 1 in every 4x4 = 1/16 pixels per image

        fprintf('Scanning %d of %d SSDD images for empirical c2 statistics (sli=%d, guard=%d)...\n', ...
            MAX_IMAGES, numel(imgs), sli_lut, guard_lut);

        idx_list = round(linspace(1, numel(imgs), MAX_IMAGES));
        for ii = 1:numel(idx_list)
            img_idx = idx_list(ii);
            I_rgb = imread(fullfile(jpeg_dir, imgs(img_idx).name));
            if ndims(I_rgb) == 3
                I_gray = rgb2gray(I_rgb);
            else
                I_gray = I_rgb;
            end
            I_img = double(I_gray);

            [~, ~, ~, stats_tmp] = WeibullCFAR_Floating(I_img, sli_lut, guard_lut, Pfa_lut, ...
                'CMin', C_MIN, 'CMax', C_MAX);
            c2map = stats_tmp.C2Map(1:PIXEL_STRIDE:end, 1:PIXEL_STRIDE:end);
            c2_samples = [c2_samples; c2map(:)]; %#ok<AGROW>
        end
        fprintf('Collected %d c2 samples from %d images.\n', numel(c2_samples), MAX_IMAGES);
    end
else
    warning(['SSDD dataset not found at %s -- using the analytical range only.\n' ...
             'Re-run this script once the dataset is reachable to refine the range empirically.'], dataset_dir);
end

if ~isempty(c2_samples)
    prc = prctile(c2_samples, [0.1 1 50 99 99.9]);
    fprintf('Empirical c2 percentiles [0.1 1 50 99 99.9] = %.6f  %.6f  %.6f  %.6f  %.6f\n', prc);
    fprintf('Empirical c2 [min, max] = [%.6f, %.6f]\n', min(c2_samples), max(c2_samples));

    % Robust (percentile-based) empirical range, immune to a handful of
    % extreme outlier windows, plus a safety margin; unioned with the
    % analytical core range so the clamp boundaries themselves are never
    % lost even if the empirical sample happened not to reach them.
    MARGIN = 0.25;   % 25% safety margin
    c2_lo_emp = prc(1)   * (1 - MARGIN);
    c2_hi_emp = prc(end) * (1 + MARGIN);

    c2_lut_min = max(1e-6, min(c2_lo_analytical, c2_lo_emp));
    c2_lut_max = max(c2_hi_analytical, c2_hi_emp);
else
    % No dataset available in this environment -- fall back to the
    % analytical range with a generous margin. Re-run once the dataset is
    % reachable so the range reflects real data, not just the math.
    c2_lut_min = c2_lo_analytical * 0.5;
    c2_lut_max = c2_hi_analytical * 1.5;
end
fprintf('Selected C-LUT input range: c2 in [%.6f, %.6f]\n', c2_lut_min, c2_lut_max);

% ---- 4c. Build the LUT: uniform quantization of c2 ---------------------
% c2 is quantized UNIFORMLY (not e.g. log(c2)) because that is what the
% streaming hardware can form cheaply: c2 arrives from the window_sum
% block as a fixed-point value already, and mapping it to a LUT address is
% then just "subtract c2_lut_min, multiply by a fixed scale, saturate,
% truncate" -- a shift/scale, no division. The trade-off is that
% C = sqrt(PSI11/c2) is quite non-linear, so quantization error in C is
% NOT uniform across the table (worst near small c2, i.e. large C); that
% error is measured explicitly below, which is also why 1024 entries were
% rejected in favour of a larger table (see the size-vs-error comparison
% printed below).
N_C_LUT_ENTRIES = 4096;    % 12-bit address; 4096x16-bit = 64 Kbit ROM,
                            % a small fraction of a Cyclone IV's embedded
                            % memory -- confirm against your specific
                            % device's M9K/M10K budget.
c2_grid = linspace(c2_lut_min, c2_lut_max, N_C_LUT_ENTRIES)';

C_raw_grid     = sqrt(PSI11 ./ c2_grid);
C_clamped_grid = min(max(C_raw_grid, C_MIN), C_MAX);

% ---- Output fixed-point format ----
% C is always in [C_MIN, C_MAX] = [0.8, 8.0], strictly positive -> an
% UNSIGNED format is used (no sign bit needed). 4 integer bits cover up to
% 15.999..., comfortably above 8.0.
CLUT_INT_BITS   = 4;
CLUT_FRAC_BITS  = 12;
CLUT_TOTAL_BITS = CLUT_INT_BITS + CLUT_FRAC_BITS;   % = 16, unsigned
CLUT_SCALE      = 2^CLUT_FRAC_BITS;

c_lut_fixed_int = round(C_clamped_grid * CLUT_SCALE);
max_code_u = 2^CLUT_TOTAL_BITS - 1;
assert(all(c_lut_fixed_int >= 0) && all(c_lut_fixed_int <= max_code_u), ...
    'c_lut: fixed-point overflow -- widen CLUT_INT_BITS.');

c_lut_requantized = c_lut_fixed_int / CLUT_SCALE;

% ---- 4d. Quantization error analysis ------------------------------------
% Error a real streaming lookup would see for a c2 value falling BETWEEN
% two grid points (nearest-neighbour LUT read), vs. the exact ("infinite
% precision", still clamped) C for that c2:
c2_dense      = linspace(c2_lut_min, c2_lut_max, 20000)';
C_ideal_dense = min(max(sqrt(PSI11 ./ c2_dense), C_MIN), C_MAX);
grid_idx = round((c2_dense - c2_lut_min) / (c2_lut_max - c2_lut_min) * (N_C_LUT_ENTRIES-1)) + 1;
grid_idx = min(max(grid_idx, 1), N_C_LUT_ENTRIES);
C_lut_dense   = c_lut_requantized(grid_idx);
c_error_dense = C_lut_dense - C_ideal_dense;

fprintf('C-LUT quantization error (LUT output vs. ideal floating-point, clamped C):\n');
fprintf('  max |error| : %.6f\n', max(abs(c_error_dense)));
fprintf('  RMS error   : %.6f\n', sqrt(mean(c_error_dense.^2)));
[worst_err, worst_i] = max(abs(c_error_dense)); %#ok<ASGLU>
fprintf('  worst case at c2=%.6f: C_ideal=%.4f, C_lut=%.4f\n', ...
    c2_dense(worst_i), C_ideal_dense(worst_i), C_lut_dense(worst_i));

% Table size vs. error, for reference / to justify N_C_LUT_ENTRIES above:
fprintf('  (table-size sweep, for reference)\n');
for N_test = [256 1024 2048 4096 8192]
    grid_test = linspace(c2_lut_min, c2_lut_max, N_test)';
    C_test = min(max(sqrt(PSI11./grid_test), C_MIN), C_MAX);
    idx_test = round((c2_dense - c2_lut_min)/(c2_lut_max-c2_lut_min)*(N_test-1)) + 1;
    idx_test = min(max(idx_test,1), N_test);
    err_test = C_test(idx_test) - C_ideal_dense;
    fprintf('    N=%5d (%2d-bit addr): max|err|=%.5f  rms=%.6f\n', ...
        N_test, ceil(log2(N_test)), max(abs(err_test)), sqrt(mean(err_test.^2)));
end

save(fullfile(lut_dir, 'c_lut.mat'), 'c2_grid', 'C_raw_grid', 'C_clamped_grid', ...
    'c_lut_fixed_int', 'c2_lut_min', 'c2_lut_max', 'C_MIN', 'C_MAX', ...
    'CLUT_INT_BITS', 'CLUT_FRAC_BITS', 'CLUT_SCALE');

fid = fopen(fullfile(lut_dir, 'c_lut.txt'), 'w');
fprintf(fid, '%% c_lut : quantized c2 -> Weibull shape parameter C, clamped to [%.2f, %.2f]\n', C_MIN, C_MAX);
fprintf(fid, '%% c2 range: [%.6f, %.6f], %d entries, uniform step = %.8f\n', ...
    c2_lut_min, c2_lut_max, N_C_LUT_ENTRIES, c2_grid(2)-c2_grid(1));
fprintf(fid, '%% Format: unsigned Q%d.%d (%d-bit), scale = 2^%d = %d\n', ...
    CLUT_INT_BITS, CLUT_FRAC_BITS, CLUT_TOTAL_BITS, CLUT_FRAC_BITS, CLUT_SCALE);
fprintf(fid, '%% Columns: index  c2  C_double(clamped)  fixed_point_code(dec)  requantized_C\n');
for k = 1:N_C_LUT_ENTRIES
    fprintf(fid, '%5d  %.8f  %.6f  %6d  %.6f\n', k-1, c2_grid(k), C_clamped_grid(k), ...
        c_lut_fixed_int(k), c_lut_requantized(k));
end
fclose(fid);

write_hex_lut(fullfile(lut_dir, 'c_lut.hex'), c_lut_fixed_int, CLUT_TOTAL_BITS, false);

fig = figure('Visible', 'off');
plot(c2_dense, C_ideal_dense, 'b-', 'LineWidth', 1.5); hold on;
plot(c2_dense, C_lut_dense,  'r--', 'LineWidth', 1);
xlabel('c2 (log-amplitude sample variance)');
ylabel('Weibull shape parameter C');
title(sprintf('C-LUT: exact (clamped) vs. quantized, N=%d entries', N_C_LUT_ENTRIES));
legend('Exact (floating point, clamped)', 'Quantized LUT (nearest-neighbour)', 'Location', 'northeast');
grid on;
saveas(fig, fullfile(lut_dir, 'c_lut_c2_vs_C.png'));
close(fig);

fprintf('  Wrote c_lut.mat / .txt / .hex / c_lut_c2_vs_C.png to %s\n\n', lut_dir);

%% =========================================================================
%% STEP 5: PFA-CONSTANT (K) LUT  —  Pfa selector -> K(Pfa)
%% =========================================================================
fprintf('=== Step 5: Pfa constant (K) LUT ===\n');

Pfa_options = [1e-3, 1e-4, 1e-5, 1e-6];
K_values = EULER_GAMMA + log(-log(Pfa_options));

fprintf('Pfa options and K(Pfa) = gamma + log(-log(Pfa)):\n');
for k = 1:numel(Pfa_options)
    fprintf('  addr %d : Pfa = %-8g -> K = %+.8f\n', k-1, Pfa_options(k), K_values(k));
end

% ---- Fixed-point format ----
% K only takes 4 known values for this Pfa set (~2.51 to ~3.20), all
% positive -- but a SIGNED format is used so the table stays valid if a
% future Pfa choice very close to 1 pushes K negative.
% 16-bit signed Q3.12: 1 sign + 3 integer bits + 12 fractional bits
%   representable range : [-8, 8 - 2^-12], resolution 2^-12 = 2.441e-4
KLUT_INT_BITS   = 3;
KLUT_FRAC_BITS  = 12;
KLUT_TOTAL_BITS = 1 + KLUT_INT_BITS + KLUT_FRAC_BITS;   % = 16
KLUT_SCALE      = 2^KLUT_FRAC_BITS;

K_fixed_int = round(K_values * KLUT_SCALE);
max_code = 2^(KLUT_TOTAL_BITS-1) - 1;
min_code = -2^(KLUT_TOTAL_BITS-1);
assert(all(K_fixed_int <= max_code) && all(K_fixed_int >= min_code), ...
    'pfa_constant_lut: fixed-point overflow -- widen KLUT_INT_BITS.');

K_requantized = K_fixed_int / KLUT_SCALE;
fprintf('  Format: signed Q%d.%d (%d-bit). Max quantization error: %.6e\n', ...
    KLUT_INT_BITS, KLUT_FRAC_BITS, KLUT_TOTAL_BITS, max(abs(K_requantized(:)' - K_values)));

save(fullfile(lut_dir, 'pfa_constant_lut.mat'), 'Pfa_options', 'K_values', 'K_fixed_int', ...
    'KLUT_INT_BITS', 'KLUT_FRAC_BITS', 'KLUT_SCALE');

fid = fopen(fullfile(lut_dir, 'pfa_constant_lut.txt'), 'w');
fprintf(fid, '%% pfa_constant_lut : Pfa selector -> K(Pfa) = gamma + log(-log(Pfa))\n');
fprintf(fid, '%% Format: signed Q%d.%d (%d-bit), scale = 2^%d = %d\n', ...
    KLUT_INT_BITS, KLUT_FRAC_BITS, KLUT_TOTAL_BITS, KLUT_FRAC_BITS, KLUT_SCALE);
fprintf(fid, '%% Columns: address  Pfa  K_double  fixed_point_code(dec)  requantized_K\n');
for k = 1:numel(Pfa_options)
    fprintf(fid, '%d  %-10g  %+.8f  %6d  %+.8f\n', k-1, Pfa_options(k), K_values(k), ...
        K_fixed_int(k), K_requantized(k));
end
fclose(fid);

write_hex_lut(fullfile(lut_dir, 'pfa_constant_lut.hex'), K_fixed_int, KLUT_TOTAL_BITS, true);
fprintf('  Wrote pfa_constant_lut.mat / .txt / .hex to %s\n\n', lut_dir);

%% =========================================================================
%% STEP 5b: COMBINED K/C LUT  —  {Pfa_sel, c2_addr} -> K(Pfa)/C  (removes divider)
%% =========================================================================
fprintf('=== Step 5b: Combined K/C LUT (removes runtime divider) ===\n');

% Reuses the exact same c2 grid/addressing as the C-LUT (Step 4c) and the
% same K(Pfa) values as Step 5. Precomputes K/C for every (Pfa, c2_addr)
% pair the hardware could ever look up -- a single flat ROM, addressed as
%   flat_addr = (Pfa_index-1)*N_C_LUT_ENTRIES + c2_addr
% i.e. Pfa forms the HIGH address bits, c2_addr the LOW bits.
N_PFA = numel(Pfa_options);
KC_double = zeros(N_C_LUT_ENTRIES, N_PFA);
for p = 1:N_PFA
    KC_double(:,p) = K_values(p) ./ C_clamped_grid;   % same C grid as c_lut
end

kc_min = min(KC_double(:));
kc_max = max(KC_double(:));
fprintf('K/C range across all %d Pfa options and %d C-LUT entries: [%.6f, %.6f]\n', ...
    N_PFA, N_C_LUT_ENTRIES, kc_min, kc_max);

% ---- Fixed-point format ----
% K/C is always positive (K>0 for all supported Pfa<1/e, C>0) -> unsigned.
KCLUT_INT_BITS  = ceil(log2(kc_max)) + 1;   % headroom above observed max
KCLUT_FRAC_BITS = 12;
KCLUT_TOTAL_BITS = KCLUT_INT_BITS + KCLUT_FRAC_BITS;
KCLUT_SCALE = 2^KCLUT_FRAC_BITS;

KC_fixed_int = round(KC_double * KCLUT_SCALE);
max_code_u = 2^KCLUT_TOTAL_BITS - 1;
assert(all(KC_fixed_int(:) >= 0) && all(KC_fixed_int(:) <= max_code_u), ...
    'kc_lut: fixed-point overflow -- widen KCLUT_INT_BITS.');

KC_requant = KC_fixed_int / KCLUT_SCALE;
kc_error = KC_requant - KC_double;
fprintf('  Format: unsigned Q%d.%d (%d-bit). Max quantization error: %.6e (resolution %.6e)\n', ...
    KCLUT_INT_BITS, KCLUT_FRAC_BITS, KCLUT_TOTAL_BITS, max(abs(kc_error(:))), 1/KCLUT_SCALE);

% Flatten in {Pfa_sel (high bits), c2_addr (low bits)} order, column-major
% reshape gives exactly flat_addr = (p-1)*N_C_LUT_ENTRIES + (idx-1):
KC_flat = reshape(KC_fixed_int, N_C_LUT_ENTRIES*N_PFA, 1);

save(fullfile(lut_dir, 'kc_lut.mat'), 'KC_double', 'KC_fixed_int', 'KC_flat', ...
    'Pfa_options', 'N_C_LUT_ENTRIES', 'KCLUT_INT_BITS', 'KCLUT_FRAC_BITS', 'KCLUT_SCALE');

fid = fopen(fullfile(lut_dir, 'kc_lut.txt'), 'w');
fprintf(fid, '%% kc_lut : K(Pfa)/C, addressed by {Pfa_sel[1:0], c2_addr[11:0]}\n');
fprintf(fid, '%% Pfa_sel 0..%d maps to Pfa_options = [%s]\n', N_PFA-1, num2str(Pfa_options));
fprintf(fid, '%% Format: unsigned Q%d.%d (%d-bit), scale = 2^%d = %d\n', ...
    KCLUT_INT_BITS, KCLUT_FRAC_BITS, KCLUT_TOTAL_BITS, KCLUT_FRAC_BITS, KCLUT_SCALE);
fprintf(fid, '%% Columns: flat_addr  pfa_sel  c2_addr  K_over_C_double  fixed_point_code(dec)\n');
for p = 1:N_PFA
    for k = 1:N_C_LUT_ENTRIES
        flat_addr = (p-1)*N_C_LUT_ENTRIES + (k-1);
        fprintf(fid, '%6d  %d  %5d  %.6f  %6d\n', flat_addr, p-1, k-1, ...
            KC_double(k,p), KC_fixed_int(k,p));
    end
end
fclose(fid);

write_hex_lut(fullfile(lut_dir, 'kc_lut.hex'), KC_flat, KCLUT_TOTAL_BITS, false);
fprintf('  Wrote kc_lut.mat / .txt / .hex to %s (%d entries, %d-bit addr, %.1f Kbit ROM)\n\n', ...
    lut_dir, numel(KC_flat), ceil(log2(numel(KC_flat))), numel(KC_flat)*KCLUT_TOTAL_BITS/1024);

fprintf('=== LUT generation complete. Files written to %s ===\n', lut_dir);


%% ===========================================================================
function write_hex_lut(filepath, fixed_int_values, nbits, is_signed)
%WRITE_HEX_LUT  Write a $readmemh / Quartus "Hex File"-compatible LUT: one
%   hex value per line, zero-padded to ceil(nbits/4) hex digits, two's-
%   complement encoding for signed values.
    nhex = ceil(nbits / 4);
    fid = fopen(filepath, 'w');
    for k = 1:numel(fixed_int_values)
        v = double(fixed_int_values(k));
        if is_signed && v < 0
            v = v + 2^nbits;
        end
        fprintf(fid, '%0*X\n', nhex, v);
    end
    fclose(fid);
end
