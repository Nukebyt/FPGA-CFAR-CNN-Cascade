function info = build_backend_lut(lut_dir, name, addr, PfaList, termFcn, f, extraMeta)
%BUILD_BACKEND_LUT  Build a {Pfa_sel, shape_addr} -> value ROM.
%
%   info = BUILD_BACKEND_LUT(lut_dir, name, addr, PfaList, termFcn, f, meta)
%
%   This is the table that removes every runtime divider from the datapath.
%   Each detector's back-end offset factorises as
%
%       delta = termFcn(shape_address_variable, Pfa)  [ * sqrt(c2) ]
%
%   and this function precomputes the bracketed term for every (Pfa,
%   address) pair the hardware could ever look up. Whatever reciprocal
%   appears in the algebra -- 1/C for Weibull, 1/v for Generalized Gamma,
%   1/rho for Burr XII -- is evaluated HERE, at generation time, in double
%   precision, and never at run time.
%
%   ADDRESS LAYOUT
%     flat_addr = (Pfa_index-1)*entries + shape_addr
%   i.e. Pfa forms the HIGH address bits and the shape variable the LOW bits,
%   matching the existing Weibull kc_lut convention so the RTL address
%   arithmetic carries over unchanged.
%
%   INPUTS
%     addr    : struct with fields
%                 .var      name of the address variable, for the header
%                 .min,.max address range (in the variable's own units)
%                 .entries  table depth
%                 .log      true to space the grid logarithmically
%               The range should come from the detector's SUPPORT CONDITION
%               (FINDINGS F4), not from the empirical spread -- see the note
%               in fixedpoint_config.m.
%     termFcn : @(addr_values, Pfa) -> column of doubles, same size
%     f       : format struct from fixedpoint_config
%
%   The function CHECKS that termFcn produced finite values everywhere and
%   reports how much of the table each Pfa plane actually uses, so a range
%   that is mostly saturated shows up at generation time rather than as a
%   mysterious detection loss later.

if nargin < 7, extraMeta = struct(); end

n = addr.entries;
if isfield(addr,'log') && addr.log
    if addr.min <= 0
        error('build_backend_lut:BadLogRange', ...
            '%s: log-spaced address needs min > 0 (got %g).', name, addr.min);
    end
    grid = logspace(log10(addr.min), log10(addr.max), n)';
else
    grid = linspace(addr.min, addr.max, n)';
end

nP  = numel(PfaList);
tbl = zeros(n, nP);

for ip = 1:nP
    v = termFcn(grid, PfaList(ip));
    v = v(:);
    if numel(v) ~= n
        error('build_backend_lut:BadTermSize', ...
            '%s: termFcn returned %d values for %d addresses.', name, numel(v), n);
    end
    bad = ~isfinite(v);
    if any(bad)
        error('build_backend_lut:NonFinite', ...
            ['%s: termFcn produced %d non-finite values at Pfa=%g ' ...
             '(first at address %d, variable %s = %.6g).\n' ...
             '  A ROM cannot hold NaN -- narrow the address range to the ' ...
             'support condition, or clamp inside termFcn.'], ...
            name, sum(bad), PfaList(ip), find(bad,1), addr.var, grid(find(bad,1)));
    end
    tbl(:,ip) = v;
end

% Column-major flatten gives exactly flat_addr = (ip-1)*n + (i-1).
flat = tbl(:);

meta = struct( ...
    'addr_var', addr.var, 'addr_min', addr.min, 'addr_max', addr.max, ...
    'entries_per_pfa', n, 'n_pfa', nP, 'Pfa_options', PfaList, ...
    'log_spaced', isfield(addr,'log') && addr.log);
fn = fieldnames(extraMeta);
for i = 1:numel(fn), meta.(fn{i}) = extraMeta.(fn{i}); end

info = lut_write(lut_dir, name, flat, f, meta);
info.grid  = grid;
info.table = tbl;
info.PfaList = PfaList;

% ---- Report the realised span per Pfa plane -----------------------------
fprintf('      address: %s in [%.6g, %.6g]%s, %d entries x %d Pfa\n', ...
    addr.var, addr.min, addr.max, ...
    tern(isfield(addr,'log') && addr.log, ' (log-spaced)', ''), n, nP);
for ip = 1:nP
    fprintf('      Pfa=%-8g value range [%+.4f, %+.4f]\n', ...
        PfaList(ip), min(tbl(:,ip)), max(tbl(:,ip)));
end
end

function o = tern(c,a,b)
    if c, o = a; else, o = b; end
end
