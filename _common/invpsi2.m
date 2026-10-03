function Y=invpsi2(X,d)
% Y = INVPSI(X,d)
%
% Inverse digamma (psi) function.  The digamma function is the
% derivative of the log gamma function.  This calculates the value
% Y > 0 for a value X such that digamma(Y) = X. Setting d to 1 or 2 
% gives the inverse of the trigamma and tetragamma functions.
%
% This algorithm is from Paul Fackler:
% http://www4.ncsu.edu/~pfackler/
%
if nargin<2, d=0; end
if ~isscalar(X)
  error('X must be scalar')
end
switch d
  case 0
    Y=exp(X);
  case 1
    Y=1./X.^2;
  case 2
    Y=min(1,1./X.^2);
  case 3
    Y=min(2,1./X.^2);
  otherwise
    error('not implemented - need to find good startig values')
end

for i=1:1000
  e=X-psi(d,Y);
  if abs(e)<1e-14, return; end
  Y=Y+e./psi(d+1,Y);
  Y=max(Y,eps);
end

return
  L = 1;
  Y = exp(X);
  while L > 10e-8
    Y = Y + L*sign(X-psi(Y));
    L = L / 2;
  end
  
  
