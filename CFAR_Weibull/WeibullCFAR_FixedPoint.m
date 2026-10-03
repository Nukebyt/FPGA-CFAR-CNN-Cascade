function [detection_map, T_log, fe, stats] = ...
    WeibullCFAR_FixedPoint(I, sli, guard, PfaIndex, lut_root)
%WEIBULLCFAR_FIXEDPOINT  Bit-accurate fixed-point Weibull CFAR.
%
%   [detection_map, T_log, fe, stats] = ...
%       WeibullCFAR_FixedPoint(I, sli, guard, PfaIndex, lut_root)
%
%   The regression anchor for the fixed-point phase. Uses ONLY:
%     * log_amp_lut        (shared, 256 entries)
%     * weibull_delta_lut  ({Pfa_sel, c2_addr} -> delta)
%     * integer add/subtract/multiply and reciprocal-multiply by constants
%   There is no log, sqrt, exp, fractional power or division anywhere in the
%   run-time path.
%
%   INPUTS
%     I        : HxW image, 8-bit-valued
%     sli,guard: geometry -- must match what the LUTs were generated for
%     PfaIndex : 1..4, selects Pfa from the table's Pfa_options
%     lut_root : the lut/ folder (default: <repo>/lut)
%
%   OUTPUTS
%     detection_map : HxW logical
%     T_log         : HxW dequantised threshold
%     fe            : the fixed-point front-end struct (for diagnostics)
%     stats         : struct incl. Pfa_used, address saturation, overflow counts
%
%   NOTE ON NAMING: the file WeibullCFAR_FixedPoint.m carried over from the
%   original project is preserved at legacy/WeibullCFAR_FixedPoint.m. That one
%   models the ORIGINAL datapath (uncentred Q2.13 log-amplitude, separate
%   c_lut and kc_lut, no third moment). This is the new shared-front-end
%   version: centred Q1.14 log-amplitude, one combined delta table, and a c3
%   accumulator chain that Weibull itself does not use but that the shared
%   front end computes for the three-parameter detectors.

if nargin < 5 || isempty(lut_root)
    paths = cfar_setup();
    lut_root = fullfile(paths.root, 'lut');
end

shared_dir  = fullfile(lut_root, 'shared');
weibull_dir = fullfile(lut_root, 'weibull');

lut = load(fullfile(weibull_dir, 'weibull_delta_lut.mat'));

if lut.sli ~= sli || lut.guard ~= guard
    error('WeibullCFAR_FixedPoint:GeometryMismatch', ...
        ['The LUTs were generated for sli=%d/guard=%d but this call uses ' ...
         '%d/%d. c2''s distribution depends on the reference-cell count, so ' ...
         'the table''s address range is not valid for a different geometry. ' ...
         'Re-run generate_LUTs_Weibull for this geometry.'], ...
        lut.sli, lut.guard, sli, guard);
end

cfg = fixedpoint_config(sli, guard);

%% ---- Shared fixed-point front end --------------------------------------
fe = cfar_front_end_fixed(I, sli, guard, shared_dir, 'Config', cfg);

%% ---- Back end: one ROM read, one add, one compare ----------------------
% Weibull has no support condition -- every c2 > 0 gives a valid C -- so the
% valid mask is all-true. (The other four detectors pass a real mask here.)
out = cfar_fixed_decide(fe, lut, fe.c2, PfaIndex, []);

detection_map = out.detection;
T_log = out.T_log;

%% ---- Stats --------------------------------------------------------------
stats = struct();
stats.Detector      = 'Weibull';
stats.Pfa_used      = lut.Pfa_options(PfaIndex);
stats.NumDetections = sum(detection_map(:));
stats.AddrSatLow    = out.SatLow;
stats.AddrSatHigh   = out.SatHigh;
stats.AddrSatHighPct= 100 * out.SatHigh / numel(detection_map);
stats.Overflow      = fe.Overflow;
stats.MeanDelta     = mean(out.delta(:));
end
