function info = lut_write(lut_dir, name, values_double, f, meta)
%LUT_WRITE  Quantise a table and write it as .mat / .txt / .hex.
%
%   info = LUT_WRITE(lut_dir, name, values_double, f, meta)
%
%   values_double : the exact (double-precision) table contents
%   f             : a format struct from fixedpoint_config (int/frac/signed/...)
%   meta          : struct of extra fields to save into the .mat and describe
%                   in the .txt header (address ranges, source expression, ...)
%
%   Writes:
%     <name>.mat  MATLAB arrays, for the fixed-point model and re-plotting
%     <name>.txt  human-readable, one row per entry, with a full header
%     <name>.hex  $readmemh / Quartus "Hex File": one value per line,
%                 zero-padded, two's complement for signed formats
%
%   ---------------------------------------------------------------------
%   SATURATION IS AN ERROR, NOT A CLAMP
%   ---------------------------------------------------------------------
%   If any value does not fit the requested format this function RAISES
%   rather than silently saturating. A table that quietly clips its own
%   contents is the kind of defect that survives to the board and then
%   presents as "the detector goes deaf above some clutter level". Callers
%   that legitimately want saturation must clamp the values themselves,
%   visibly, before calling.
%
%   The returned info includes the realised quantisation error so each
%   generator can report it and the verification gate can check it.

if nargin < 5, meta = struct(); end
if ~isfolder(lut_dir), mkdir(lut_dir); end

v = values_double(:);
if ~all(isfinite(v))
    error('lut_write:NonFinite', ...
        '%s: %d of %d entries are not finite. Fix the generator -- a NaN in a ROM is not recoverable at run time.', ...
        name, sum(~isfinite(v)), numel(v));
end

code = round(v * f.scale);

if f.signed
    lo = -2^(f.total-1);
    hi =  2^(f.total-1) - 1;
else
    lo = 0;
    hi =  2^f.total - 1;
end
if any(code < lo) || any(code > hi)
    bad = find(code < lo | code > hi);
    error('lut_write:Overflow', ...
        ['%s: %d of %d entries overflow Q%d.%d (%s, %d-bit).\n' ...
         '  worst value %.6g needs more than %d integer bits.\n' ...
         '  Widen the format in fixedpoint_config.m, or clamp deliberately before calling.'], ...
        name, numel(bad), numel(v), f.int, f.frac, tern(f.signed,'signed','unsigned'), ...
        f.total, v(bad(abs(v(bad)) == max(abs(v(bad))))), f.int);
end

requant = code / f.scale;
err     = requant - v;

%% ---- .mat ---------------------------------------------------------------
S = struct();
S.name        = name;
S.values      = v;
S.code        = code;
S.requantized = requant;
S.format      = f;
S.maxAbsError = max(abs(err));
S.rmsError    = sqrt(mean(err.^2));
fn = fieldnames(meta);
for i = 1:numel(fn)
    S.(fn{i}) = meta.(fn{i});
end
save(fullfile(lut_dir, [name '.mat']), '-struct', 'S');

%% ---- .txt ---------------------------------------------------------------
fid = fopen(fullfile(lut_dir, [name '.txt']), 'w');
fprintf(fid, '%% %s : %d entries\n', name, numel(v));
fprintf(fid, '%% Format: %s Q%d.%d (%d-bit), scale = 2^%d = %d, resolution = %.6e\n', ...
    tern(f.signed,'signed','unsigned'), f.int, f.frac, f.total, f.frac, f.scale, f.res);
fprintf(fid, '%% Quantisation error: max |e| = %.6e, RMS = %.6e\n', S.maxAbsError, S.rmsError);
for i = 1:numel(fn)
    val = meta.(fn{i});
    if isnumeric(val) && isscalar(val)
        fprintf(fid, '%% %s = %.10g\n', fn{i}, val);
    elseif ischar(val)
        fprintf(fid, '%% %s = %s\n', fn{i}, val);
    elseif isnumeric(val) && numel(val) <= 8
        fprintf(fid, '%% %s = %s\n', fn{i}, mat2str(val, 8));
    end
end
fprintf(fid, '%% Columns: addr  exact_value  code(dec)  requantized  error\n');
for i = 1:numel(v)
    fprintf(fid, '%6d  %+.10f  %8d  %+.10f  %+.3e\n', ...
        i-1, v(i), code(i), requant(i), err(i));
end
fclose(fid);

%% ---- .hex ---------------------------------------------------------------
nhex = ceil(f.total / 4);
fid = fopen(fullfile(lut_dir, [name '.hex']), 'w');
for i = 1:numel(code)
    c = double(code(i));
    if f.signed && c < 0
        c = c + 2^f.total;          % two's complement
    end
    fprintf(fid, '%0*X\n', nhex, c);
end
fclose(fid);

%% ---- info ---------------------------------------------------------------
info = struct();
info.name        = name;
info.entries     = numel(v);
info.bits        = f.total;
info.kbit        = numel(v) * f.total / 1024;
info.maxAbsError = S.maxAbsError;
info.rmsError    = S.rmsError;
info.code        = code;
info.requantized = requant;

fprintf('    %-22s %6d x %2d bit = %7.1f Kbit | max|err| %.3e  rms %.3e\n', ...
    name, info.entries, info.bits, info.kbit, info.maxAbsError, info.rmsError);
end


function o = tern(c, a, b)
    if c, o = a; else, o = b; end
end
