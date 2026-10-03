function [des]=matplot2(X,Y,A,r)
% This function displays a matrix within the 
% dynamic range of r

% This function is similar to matplot.m except the location of the origin
 
% Inputs:
% A : the matrix 
% r : dynamic range of the display [dB]
% X : x-label Vector
% Y : y-label vector
 
% Output:
% p : matrix thresholded to r(dB)
 
b = max(max(abs(A))); %find max value of A
ra = b/(10^(r/20)); % make it to dB
 
% treshold A to the dynamic range of r[dB]

p = A.*(abs(A>=ra)+ra*ones(size(A)).*(abs(A)<ra));
pp = 20*log10(abs(p)/b); 
des=p;
colormap(jet(256))
imagesc(X,Y,pp)

% colormap(jet(256));
% imagesc(X,Y,pp)


