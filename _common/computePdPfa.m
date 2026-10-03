function [Pd, Pfa_actual, n_detected_ships, n_total_ships, n_false, n_bg_pixels] = ...
    computePdPfa(detection_map, gt_boxes, h, w)
% Pd  = fraction of ground-truth ships with >=1 detected pixel inside their box
% Pfa = false detections outside all boxes / total background pixel count

n_total_ships = size(gt_boxes, 1);
n_detected_ships = 0;

ship_mask = false(h, w);  % union of all GT boxes, for Pfa background exclusion

for k = 1:n_total_ships
    xmin = max(1, round(gt_boxes(k,1)));
    ymin = max(1, round(gt_boxes(k,2)));
    xmax = min(w, round(gt_boxes(k,3)));
    ymax = min(h, round(gt_boxes(k,4)));

    ship_mask(ymin:ymax, xmin:xmax) = true;

    box_detections = detection_map(ymin:ymax, xmin:xmax);
    if any(box_detections(:))
        n_detected_ships = n_detected_ships + 1;
    end
end

if n_total_ships > 0
    Pd = n_detected_ships / n_total_ships;
else
    Pd = NaN;
end

false_map = detection_map & ~ship_mask;
n_false = sum(false_map(:));
n_bg_pixels = sum(~ship_mask(:));
Pfa_actual = n_false / n_bg_pixels;
end