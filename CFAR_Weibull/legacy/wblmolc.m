function [b, c] = wblmolc(c1, c2)
    c = sqrt(psi(1,1) / c2);
    b = exp(c1 - (psi(1) / c));
end