function boxMap = load_coco_boxes(json_path)
%LOAD_COCO_BOXES  Parse an MS-COCO-format annotation JSON once and return a
%   containers.Map from image file_name -> ground-truth boxes, in the same
%   [xmin ymin xmax ymax] layout parseVOCBoxes.m returns for SSDD, so
%   cfar_metrics stays dataset-agnostic.
%
%   boxMap = LOAD_COCO_BOXES(json_path)
%
%   Parsing the whole file once (instead of per-image) matters here: HRSID's
%   train_test2017.json holds all 5604 images and 16951 annotations in one
%   file, and re-parsing it per image inside a sweep loop would dominate
%   runtime for no benefit -- the whole point of the front-end/estimator
%   factoring in run_comparison.m.
%
%   Images with zero annotated ships still get a map entry (0x4 empty), so a
%   caller can tell "no ships in this image" apart from "file not in the
%   annotation set at all".
    raw = fileread(json_path);
    d = jsondecode(raw);

    imgs = d.images;
    nImg = numel(imgs);
    id2name = containers.Map('KeyType', 'double', 'ValueType', 'char');
    boxMap  = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for i = 1:nImg
        id2name(imgs(i).id) = imgs(i).file_name;
        boxMap(imgs(i).file_name) = zeros(0, 4);
    end

    anns = d.annotations;
    nAnn = numel(anns);
    for i = 1:nAnn
        a = anns(i);
        name = id2name(a.image_id);
        % COCO bbox is [x, y, width, height] (top-left corner).
        b = a.bbox(:)';
        xyxy = [b(1), b(2), b(1)+b(3), b(2)+b(4)];
        boxMap(name) = [boxMap(name); xyxy];
    end
end
