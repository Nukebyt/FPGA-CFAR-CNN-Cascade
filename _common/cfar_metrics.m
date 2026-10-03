function m = cfar_metrics(detection_map, gt_boxes, valid_mask)
%CFAR_METRICS  Detection-quality metrics for one CFAR output against
%   VOC-style ground-truth boxes.
%
%   m = CFAR_METRICS(detection_map, gt_boxes)
%   m = CFAR_METRICS(detection_map, gt_boxes, valid_mask)
%
%   Reports BOTH a pixel-level and an object-level view, because on this
%   problem they answer different questions and either alone is misleading:
%
%   PIXEL LEVEL
%     Pd_ship     fraction of ground-truth ships with at least one detected
%                 pixel inside their box. This is the headline number the
%                 existing Weibull project reports (computePdPfa.m), kept
%                 identical so the new results are directly comparable to
%                 its 88.7% figure.
%     Pfa_actual  detected pixels outside every box / total background
%                 pixels. Its agreement with the NOMINAL Pfa is the single
%                 best evidence that a clutter model actually fits -- a
%                 detector whose measured rate is 50x its nominal one is
%                 mis-modelling the tail, however good its Pd looks.
%
%   OBJECT LEVEL
%     Pixel Pd alone rewards a detector that lights up half the image: every
%     ship gets "detected". Object-level metrics do not. Detections are
%     grouped into 8-connected clusters, then
%       TP  ground-truth boxes hit by at least one cluster (deduped per box)
%       FP  clusters overlapping no ground-truth box
%       FN  ground-truth boxes hit by no cluster
%     giving precision, recall and F1 -- F1 is the metric used to RANK
%     detectors in Phase 2, since it is the one that penalises both misses
%     and clutter breakthrough.
%
%     BUG FIX (2026-09, see BUG_LOG.md D23): TP used to be counted PER
%     CLUSTER (any cluster overlapping a box counted as its own TP), while FN
%     was always counted per ground-truth box. A ship that fragments into
%     several 8-connected clusters -- Weibull was observed fragmenting into
%     ~8.9 clusters per found ship -- contributed several TPs against a
%     single GT box, inflating Precision/Recall/F1 for detectors that
%     fragment more. TP is now deduped per matched GT box (`sum(hit_box)`),
%     symmetric with how FN already worked. Any F1/Precision/Recall number
%     computed before this fix (including the original Phase 2 ranking in
%     FINDINGS.md sec. 7) used the old, asymmetric counting and should be
%     treated as suspect until re-run.
%
%   INVALID WINDOWS
%     valid_mask (optional) marks pixels where the estimator had no solution
%     and no threshold could be formed. Those pixels can never produce a
%     detection, so counting them as "background correctly rejected" would
%     flatter a detector that simply failed to run. They are EXCLUDED from
%     the background-pixel denominator, so Pfa_actual is measured only over
%     the area the detector actually covered. m.CoverageFraction reports how
%     much of the image that was -- it must be read alongside Pd, and is why
%     FractionInvalid is a first-class metric in the sweep.
%
%   OUTPUT (struct m)
%     Pd_ship, DetectedShips, TotalShips
%     Pfa_actual, FalsePixels, BackgroundPixels, CoverageFraction
%     TP, FP, FN, Precision, Recall, F1, NumClusters
%     NumDetections

[h, w] = size(detection_map);
if nargin < 3 || isempty(valid_mask)
    valid_mask = true(h, w);
end

detection_map = logical(detection_map);

%% ---- Ground-truth mask -------------------------------------------------
n_ships   = size(gt_boxes, 1);
ship_mask = false(h, w);
boxRC     = zeros(n_ships, 4);      % [r0 r1 c0 c1], clipped to the image

for k = 1:n_ships
    c0 = max(1, round(gt_boxes(k,1)));
    r0 = max(1, round(gt_boxes(k,2)));
    c1 = min(w, round(gt_boxes(k,3)));
    r1 = min(h, round(gt_boxes(k,4)));
    boxRC(k,:) = [r0 r1 c0 c1];
    ship_mask(r0:r1, c0:c1) = true;
end

%% ---- Pixel level -------------------------------------------------------
detected_ship = false(n_ships, 1);
for k = 1:n_ships
    b = boxRC(k,:);
    detected_ship(k) = any(any(detection_map(b(1):b(2), b(3):b(4))));
end

bg = ~ship_mask & valid_mask;        % background the detector actually covered
false_px = detection_map & bg;

m = struct();
m.TotalShips       = n_ships;
m.DetectedShips    = sum(detected_ship);
if n_ships > 0
    m.Pd_ship = m.DetectedShips / n_ships;
else
    m.Pd_ship = NaN;
end
m.FalsePixels      = sum(false_px(:));
m.BackgroundPixels = sum(bg(:));
if m.BackgroundPixels > 0
    m.Pfa_actual = m.FalsePixels / m.BackgroundPixels;
else
    m.Pfa_actual = NaN;
end
m.CoverageFraction = mean(valid_mask(:));
m.NumDetections    = sum(detection_map(:));

%% ---- Object level ------------------------------------------------------
cc = bwconncomp(detection_map, 8);
m.NumClusters = cc.NumObjects;

FP = 0;
hit_box = false(n_ships, 1);

for i = 1:cc.NumObjects
    idx = cc.PixelIdxList{i};
    hits = false;
    if n_ships > 0
        [rr, ccol] = ind2sub([h w], idx);
        for k = 1:n_ships
            b = boxRC(k,:);
            if any(rr >= b(1) & rr <= b(2) & ccol >= b(3) & ccol <= b(4))
                hit_box(k) = true;
                hits = true;
            end
        end
    end
    % FP stays per-cluster (each spurious cluster is a distinct false
    % alarm); TP is deduped per GT box below, not counted here -- a ship
    % fragmenting into N clusters must count as ONE true positive, not N.
    if ~hits, FP = FP + 1; end
end

TP = sum(hit_box);
FN = n_ships - TP;

m.TP = TP;
m.FP = FP;
m.FN = FN;
if (TP + FP) > 0, m.Precision = TP / (TP + FP); else, m.Precision = NaN; end
if (TP + FN) > 0, m.Recall    = TP / (TP + FN); else, m.Recall    = NaN; end
if isfinite(m.Precision) && isfinite(m.Recall) && (m.Precision + m.Recall) > 0
    m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall);
else
    m.F1 = 0;
end
end
