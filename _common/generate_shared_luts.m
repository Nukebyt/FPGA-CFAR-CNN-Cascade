function generate_shared_luts(varargin)
%GENERATE_SHARED_LUTS  Phase 3: the tables that are NOT detector-specific.
%
%   GENERATE_SHARED_LUTS()
%   GENERATE_SHARED_LUTS('Sli',17,'Guard',13)
%
%   Builds the ROM that ALL FIVE detectors share, into lut/shared/:
%
%     log_amp_lut : I (0..255) -> xc = log(sqrt(I+0.5)) - X0, 256 entries
%
%   The other shared tables (log2_mant_lut, sqrt_mant_lut) are built by
%   generate_logskew_luts.m, because they exist to serve the log-domain shape
%   address and belong with it.
%
%   ---------------------------------------------------------------------
%   WHERE sqrt(c2) WENT
%   ---------------------------------------------------------------------
%   Every detector's back-end offset factorises into a shape-only part and a
%   c2-only part, and in four of the five the c2-only part is exactly sqrt(c2):
%
%     Weibull      delta = [K(Pfa)/sqrt(psi1(1))]                  * sqrt(c2)
%     Lognormal    delta = [z(Pfa)]                                * sqrt(c2)
%     Gen. Gamma   delta = [Num(k,Pfa)/sqrt(psi1(k))] * sgn        * sqrt(c2)
%     Burr XII     delta = [Num(kap,Pfa)/sqrt(psi1(kap)+psi1(1))]  * sqrt(c2)
%
%   which suggested one shared sqrt(c2) ROM. In practice it split two ways:
%     * Weibull and Lognormal fold the sqrt AND their own clamp straight into
%       their delta ROM -- cheaper (no multiplier) and more accurate, because
%       the clamp absorbs the sqrt's worst region near c2 = 0 entirely.
%     * Gen. Gamma and Burr XII genuinely need sqrt(c2) as a separate factor,
%       and get it from sqrt_fixed.m (exponent/mantissa), not a uniform table.
%   See BUG_LOG D9 and section 2 below.
%
%   ---------------------------------------------------------------------
%   THE log_amp_lut CARRIES THE CENTRING AND THE DARK-PIXEL FLOOR
%   ---------------------------------------------------------------------
%   Two things are folded into this table at generation time, both free:
%     * subtraction of X0 (BUG_LOG H1 -- makes c3 viable in fixed point)
%     * the optional intensity floor (FINDINGS F6 / section 7.8 -- clamps the
%       few darkest entries, which on JPEG-encoded imagery are outlier-driven
%       rather than informative)
%   Neither costs a gate: they only change what the 256 stored values are.
%
%   NAME-VALUE OPTIONS
%     'Sli','Guard' : geometry, for the config (default 17 / 13)
%     'IFloor'      : intensity floor baked into the table (default 0 = off)
%     'Entries'     : reserved (unused since sqrt_c2_lut was removed)

p = inputParser;
addParameter(p, 'Sli',     17);
addParameter(p, 'Guard',   13);
addParameter(p, 'IFloor',  0);
addParameter(p, 'C2Max',   2.0);
addParameter(p, 'Entries', []);
parse(p, varargin{:});

paths = cfar_setup();
cfg   = fixedpoint_config(p.Results.Sli, p.Results.Guard);
lut_dir = fullfile(paths.root, 'lut', 'shared');
if ~isfolder(lut_dir), mkdir(lut_dir); end

nEnt = p.Results.Entries;
if isempty(nEnt), nEnt = cfg.N_SHAPE_ENTRIES; end

fprintf('=====================================================================\n');
fprintf(' Shared LUTs  (geometry sli=%d guard=%d, N=%d)\n', cfg.sli, cfg.guard, cfg.N);
fprintf('   output: %s\n', lut_dir);
fprintf('=====================================================================\n');

%% =====================================================================
%% 1. log_amp_lut
%% =====================================================================
fprintf('\n--- log_amp_lut : I -> log(sqrt(I+0.5)) - X0 ---\n');

Ivals = (0:255)';
Iclamped = Ivals;
if p.Results.IFloor > 0
    Iclamped = max(Ivals, p.Results.IFloor);
    fprintf('  IFloor = %d : entries 0..%d all hold the value for I=%d\n', ...
        p.Results.IFloor, p.Results.IFloor-1, p.Results.IFloor);
end
xc = log(sqrt(Iclamped + 0.5)) - cfg.X0;

fprintf('  X0 = %.4f (centring, baked in)\n', cfg.X0);
fprintf('  xc range over the table: [%+.4f, %+.4f]  (format holds +-%.4f)\n', ...
    min(xc), max(xc), cfg.xc.max);

lut_write(lut_dir, 'log_amp_lut', xc, cfg.xc, struct( ...
    'X0', cfg.X0, 'IFloor', p.Results.IFloor, ...
    'expr', 'log(sqrt(max(I,IFloor)+0.5)) - X0', ...
    'addr_min', 0, 'addr_max', 255));

%% =====================================================================
%% 2. (REMOVED) sqrt_c2_lut -- superseded by the exponent/mantissa sqrt
%% =====================================================================
% A uniformly-addressed table on c2 in [0,2] used to live here. It was REMOVED
% rather than deprecated, because a stale ROM left in lut/ is a live hazard --
% nothing stops a future model or testbench loading the wrong one.
%
% Why it was wrong: delta is PROPORTIONAL to sqrt(c2) for four of the five
% detectors (FINDINGS F2/F10), so the threshold needs uniform RELATIVE
% accuracy. A uniform address gives uniform ABSOLUTE accuracy, and sqrt is
% singular at c2 -> 0. Measured realised error 1.5e-2 worst case, barely
% responsive to depth (6.3e-2 at 256 entries, 1.1e-2 at 8192). See BUG_LOG D9.
%
% Replaced by sqrt_mant_lut + sqrt_fixed.m (built in generate_logskew_luts.m),
% which bounds the error by the mantissa step at every magnitude and reuses the
% leading-zero counter the log-domain shape address already needs.
%
% Weibull and Lognormal need no sqrt table at all: each folds the sqrt AND its
% own clamp into its delta ROM, which is cheaper and more accurate than a
% shared sqrt followed by a multiply.

fprintf('\n=== Shared LUTs written to %s ===\n', lut_dir);
end
