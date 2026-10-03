function boxes = parseVOCBoxes(xml_path)
% Parses a PASCAL-VOC-style annotation XML and returns [xmin ymin xmax ymax]
% for each <object> found.
doc = xmlread(xml_path);
objs = doc.getElementsByTagName('object');
n = objs.getLength();
boxes = zeros(n, 4);

for k = 0:n-1
    obj = objs.item(k);
    bnd = obj.getElementsByTagName('bndbox').item(0);

    xmin = str2double(bnd.getElementsByTagName('xmin').item(0).getTextContent());
    ymin = str2double(bnd.getElementsByTagName('ymin').item(0).getTextContent());
    xmax = str2double(bnd.getElementsByTagName('xmax').item(0).getTextContent());
    ymax = str2double(bnd.getElementsByTagName('ymax').item(0).getTextContent());

    boxes(k+1, :) = [xmin, ymin, xmax, ymax];
end
end