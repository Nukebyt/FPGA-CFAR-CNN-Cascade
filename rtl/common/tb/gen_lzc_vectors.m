function gen_lzc_vectors()
paths = cfar_setup();
mantLut = load(fullfile(paths.root,'lut','shared','log2_mant_lut.mat'));
sqMant  = load(fullfile(paths.root,'lut','shared','sqrt_mant_lut.mat'));

% c2 format: Q2.24 unsigned, 26 bits. Sweep magnitudes across several decades
% plus a dense linear sweep, matching what real c2/|c3| codes actually span.
codes = unique(round([logspace(0, log10(2^26-1), 400), 1:1:2000, (2^26-1)]));
codes = codes(codes>=1 & codes<=2^26-1);
codes = double(codes(:));

fracBits = 24;
lg = log2_fixed(codes, fracBits, mantLut);
sq = sqrt_fixed(codes, fracBits, sqMant);

fid = fopen(fullfile(paths.root,'rtl','common','tb','lzc_vectors.txt'),'w');
fprintf(fid, '%% code lg_code(Q5.16) sq_code(Q2.14)\n');
OUT_FRAC_LG = 16; OUT_INT_SQ = 2; OUT_FRAC_SQ = 14;
for i = 1:numel(codes)
    lg_code = round(lg(i) * 2^OUT_FRAC_LG);
    sq_code = round(sq(i) * 2^OUT_FRAC_SQ);
    fprintf(fid, '%d %d %d\n', codes(i), lg_code, sq_code);
end
fclose(fid);
fprintf('Wrote %d test vectors.\n', numel(codes));
end
