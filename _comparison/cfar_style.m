function S = cfar_style()
%CFAR_STYLE  Shared figure styling for the Phase 2 comparison plots.
%
%   S = CFAR_STYLE()
%
%   One place for the palette, marker set and axis treatment, so every figure
%   in _comparison/Figures reads as one system and a detector keeps the same
%   colour in every chart it appears in.
%
%   ---------------------------------------------------------------------
%   COLOUR
%   ---------------------------------------------------------------------
%   The hues are a validated categorical palette, assigned in FIXED slot order
%   (blue, orange, aqua, yellow, magenta, green, violet) and never cycled.
%   The five core detectors take slots 1-5; the two variants take 6-7 and are
%   plotted on separate axes rather than crowded onto the core charts. The
%   worst adjacent pair in this set separates by dE 9.1 under protanopia and
%   19.6 under normal vision -- i.e. it is colour-blind safe as a set.
%
%   Colour is never the ONLY channel: every detector also carries a distinct
%   marker shape and line style, which is what keeps these figures readable
%   printed in greyscale -- the normal fate of a report figure.
%
%   Colour follows the DETECTOR, not its rank in the current plot, so
%   filtering to a subset never repaints the survivors.
%
%   ---------------------------------------------------------------------
%   AXES
%   ---------------------------------------------------------------------
%   Grid and axis lines are recessive (light grey, thin) so the data sits in
%   front of them. Marks are thin: 1.8pt lines, 6pt markers. No chart in this
%   set uses two y-scales -- where two quantities of different scale need
%   comparing they get two panels.

S = struct();

% Fixed slot order -- do not reorder, the CVD validation depends on it.
S.colors = containers.Map( ...
    {'Weibull','Lognormal','GenGamma','G0','BurrXII','GenGamma-exact','G0-L1'}, ...
    { [42 120 214]/255, ...   % 1 blue
      [235 104  52]/255, ...   % 2 orange
      [ 27 175 122]/255, ...   % 3 aqua
      [237 161   0]/255, ...   % 4 yellow
      [232 123 164]/255, ...   % 5 magenta
      [  0 131   0]/255, ...   % 6 green
      [ 74  58 167]/255});     % 7 violet

% Secondary encoding: shape and dash, so identity survives greyscale printing.
S.markers = containers.Map( ...
    {'Weibull','Lognormal','GenGamma','G0','BurrXII','GenGamma-exact','G0-L1'}, ...
    {'o','s','^','d','v','>','<'});

S.lines = containers.Map( ...
    {'Weibull','Lognormal','GenGamma','G0','BurrXII','GenGamma-exact','G0-L1'}, ...
    {'-','-','-','-','-','--','--'});

S.core    = {'Weibull','Lognormal','GenGamma','G0','BurrXII'};
S.variant = {'GenGamma-exact','G0-L1'};

S.lw      = 1.8;
S.ms      = 6;
S.font    = 'Helvetica';
S.fontsz  = 11;
S.grid    = [0.88 0.88 0.87];
S.axcol   = [0.32 0.32 0.30];
S.ink     = [0.04 0.04 0.04];
end
