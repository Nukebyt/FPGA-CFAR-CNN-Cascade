function g = randg_local(a, m, n)
%RANDG_LOCAL  Gamma(shape a, scale 1) random numbers, without the Statistics
%   Toolbox.
%
%   g = RANDG_LOCAL(a, m, n)  returns an m-by-n array of Gamma(a,1) draws.
%
%   randg / gamrnd both require the Statistics and Machine Learning Toolbox,
%   which is not installed. This is needed only by verify_models.m, to
%   generate synthetic clutter from the generalized-gamma and G0
%   distributions so their THRESHOLD formulas (not just their estimators)
%   can be calibration-checked against a known nominal Pfa.
%
%   METHOD: Marsaglia & Tsang (2000), "A Simple Method for Generating Gamma
%   Variables", ACM TOMS 26(3). For a >= 1 it is a squeeze-accelerated
%   rejection sampler on the transformation x = d*(1 + c*N)^3 with
%   d = a - 1/3 and c = 1/sqrt(9d); acceptance is above 96% for every a >= 1,
%   so the rejection loop terminates promptly. For a < 1 the boost identity
%       Gamma(a) =d= Gamma(a+1) * U^(1/a),   U ~ Uniform(0,1)
%   is applied, which is exact.
%
%   Vectorised: the rejection loop operates on the still-unaccepted subset,
%   so the whole array is filled in a handful of passes rather than one
%   element at a time.

if nargin < 3, n = 1; end
if ~isscalar(a) || ~(a > 0)
    error('randg_local:BadShape', 'Shape a must be a positive scalar.');
end

N = m * n;

boost = 1;
if a < 1
    % exact boost identity, applied after sampling Gamma(a+1)
    boost = rand(N, 1) .^ (1/a);
    a = a + 1;
end

d = a - 1/3;
c = 1 / sqrt(9*d);

g = zeros(N, 1);
todo = (1:N)';

while ~isempty(todo)
    k = numel(todo);
    x = randn(k, 1);
    v = (1 + c*x).^3;
    u = rand(k, 1);

    % v <= 0 is an automatic reject (the cube-root transform is only valid
    % for 1 + c*x > 0).
    accept = (v > 0) & ( ...
        (u < 1 - 0.0331 * x.^4) | ...                       % fast squeeze
        (log(u) < 0.5*x.^2 + d*(1 - v + log(max(v, realmin)))) );  % exact test

    g(todo(accept)) = d * v(accept);
    todo = todo(~accept);
end

g = g .* boost;
g = reshape(g, m, n);
end
