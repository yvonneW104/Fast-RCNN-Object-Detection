function bboxes = xywh_format (bboxes)
    x1 = bboxes(:,1);
    y1 = bboxes(:,2);
    x2 = bboxes(:,3);
    y2 = bboxes(:,4);
    
    w = x2 - x1 + 1;
    h = y2 - y1 + 1;
    
    bboxes(:,3) = w;
    bboxes(:,4) = h;
end