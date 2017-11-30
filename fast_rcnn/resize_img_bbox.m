function [resized_image, resized_bboxes] = resize_img_bbox (image, bboxes)
    [height, width, ~] = size(image);
    
    % set the shorter edge to 600
    min_edge = 600;
    
    ratio = min_edge / min(height, width);
    
    resized_image = imresize(image, ratio);
    
    % first column are 1, the last 4 columns are positions of proposals
    resized_bboxes = round(ratio .* bboxes);
    resized_bboxes(:,1:2) = resized_bboxes(:,1:2) + 1;
    resized_bboxes = single([ones(size(bboxes,1),1), resized_bboxes]);
end