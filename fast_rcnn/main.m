clearvars;
close all;

% part 1
% run Setup.m

% Load data
net = preprocessNet(load('../../data/models/fast-rcnn-caffenet-pascal07-dagnn.mat'));
img_part1 = imread('../../example.jpg');
proposals = load('../../example_boxes.mat');
init_roi = single(proposals.boxes);

% Resize the image and ROIs
[resized_img, resized_ROIs] = resize_img_bbox(single(img_part1), init_roi);

% Remove the average color
resized_img = bsxfun(@minus, resized_img, net.meta.normalization.averageImage);

% Evaluate the output of the model
net.eval({'data', resized_img, 'rois', resized_ROIs.'});
probabilities = squeeze(net.getVar('cls_prob').value);
bbox_regression = squeeze(net.getVar('bbox_pred').value);

car_index = 8;
nms_thred = 0.3;

% Get the initial probabilities and bbox regression for car
init_probs_car = probabilities(car_index,:).';
bbox_reg_car = bbox_regression((car_index-1)*4+1 : car_index*4, :).';
init_bbox_reg_car = bbox_transform_inv(init_roi, bbox_reg_car);

% NMS to choose detection
chosen_detection_car = nms([init_bbox_reg_car, init_probs_car], nms_thred);
bbox_reg_car = chosen_detection_car(:, 1:4);
probs_car = chosen_detection_car(:,5);

% Plot the number of detections in the image over the value of threshold
min_thred = min(probs_car);
max_thred = max(probs_car);
num_thred = 100;
thresholds = linspace(min_thred, max_thred, num_thred);
positives = repmat(probs_car.', [num_thred,1]) > repmat(thresholds, [size(probs_car), 1]).';
num_of_detection = sum(positives,2);
chosen_threshold = 0.4;

% plot the number of detection v.s. thresholds
% figure;
% plot(thresholds, num_of_detection);
% xlabel('thresholds');
% ylabel('number of detections');
% title('Number of Detection v.s. Thresholds');

% Visualize the detected bounding boxes with probability score
detections = find(probs_car > chosen_threshold);
figure;
imshow(img_part1);
hold on
for i = 1:size(detections,1)
   x = bbox_reg_car(detections(i),1);
   y = bbox_reg_car(detections(i),2);
   w = bbox_reg_car(detections(i),3) - x + 1;
   h = bbox_reg_car(detections(i),4) - y + 1;
   rectangle('Position', [x,y,w,h], 'EdgeColor','r');
   each_prob = sprintf('%0.2f', probs_car(detections(i)));
   text(double(x),double(y), each_prob, ...
       'BackgroundColor','r','Color','b', ...
       'Margin', 1, 'FontSize',10);
end
hold off



% part 2
clearvars -except net chosen_threshold

num_imgs = 4952;
num_classes = 20;
max_selection = 100;
run_flag = 0;

net = preprocessNet(load('../../data/models/fast-rcnn-caffenet-pascal07-dagnn.mat'));
images_dir = '../../data/images';
annotation_dir = '../../data/annotations'; 
annotation_files = dir(annotation_dir);
proposals = load('../../data/SSW/SelectiveSearchVOC2007test.mat');

% Read the ground truth annotation
annotations = cell(num_imgs, 1);
training_data(num_imgs,num_classes) = struct('bboxes', []);
class_names = net.meta.classes.name(2:end);
count = 1;

% Read annotations & get training_data for evaluateDetectionPrecision function
for i = 1:numel(annotation_files)
    file = annotation_files(i);
    if file.bytes == 0
       continue; 
    end
    annotations{count} = PASreadrecord(fullfile('../../data/annotations',file.name));
    for o = annotations{count}.objects
        for c = 1:num_classes
            if strcmp(o.class,class_names{c})
                bbox = xywh_format(o.bbox) + [1,1,0,0];
                bboxes = [training_data(count,c).bboxes; bbox];
                training_data(count,c).bboxes = bboxes;   
            end
        end
    end
    count = count + 1;
end

% Get the evaluate results for 4952 images
if run_flag
    for i = 1:num_imgs
        tic
        img = imread(fullfile(images_dir, annotations{i}.filename));
        rois = single(proposals.boxes{1,i});
        [resized_img, resized_ROIs] = resize_img_bbox(single(img), rois);
        resized_img = bsxfun(@minus, resized_img, net.meta.normalization.averageImage);

        net.eval({'data', resized_img, 'rois', resized_ROIs.'});
        probabilities = squeeze(net.getVar('cls_prob').value);
        bbox_regression = squeeze(net.getVar('bbox_pred').value);

        evaluate_results(i).name = annotations{i}.filename;
        evaluate_results(i).probs = probabilities;
        evaluate_results(i).bbox_reg = bbox_regression;
        
        for class = 1:num_classes
            tic
            init_probs_class = probabilities(class+1,:).';
            bbox_reg_class = bbox_regression(class*4+1 : (class+1)*4, :).';
            init_bbox_reg_class = bbox_transform_inv(rois, bbox_reg_class);

            detection = nms([init_bbox_reg_class, init_probs_class], nms_thred);
            bbox_reg_class = detection(1: min(max_selection, size(detection,1)), 1:4);
            probs_class = detection(1: min(max_selection, size(detection,1)),5);
            
            chosen_detection{j, class} = [bbox_reg_class, probs_class];
            fprintf('%d image %d class %.4f seconds\n',i, class, toc);
        end
    end
    save('chosen_detection.mat', '-v7.3', 'chosen_detection');
else
    load('chosen_detection.mat', 'chosen_detection');
    fprintf('chosen_detection loaded\n');
end

% Draw precision-recall curve
sum_ap = 0;
if run_flag
    detection_results(num_imgs) = struct('Boxes', [], 'Scores', []);
    class_ap(num_classes) = struct('Name', [], 'ap', [], 'recall', [], 'precision', []);
    for class = 1:num_classes
        tic
        for i = 1:num_imgs
            bboxes = chosen_detection{i,class}(:, 1:4);
            scores = chosen_detection{i,class}(:, 5);
            detection_results(i).Boxes = xywh_format(bboxes) + [1,1,0,0];
            detection_results(i).Scores = scores;
        end
        [ap,recall,precision] = evaluateDetectionPrecision(struct2table(detection_results), struct2table(training_data(:,class)), chosen_threshold);
        class_ap(class).Name = class_names(class);
        class_ap(class).ap = ap;
        class_ap(class).recall = recall;
        class_ap(class).precision = precision;

        sum_ap = sum_ap + ap;
        fprintf('%d class %.4f seconds\n',class, toc);
    end
    save('class_ap.mat', '-v7.3', 'class_ap');
else    
    load('class_ap.mat', 'class_ap');
    for class = 1:num_classes
        sum_ap = sum_ap + class_ap(class).ap;
    end
end

MAP = sum_ap / num_classes;

% figure;
% plot(class_ap(car_index).recall, class_ap(car_index).precision)
% xlabel('Recall');
% ylabel('Precision');

% Get the index of image contains most true positive detections of multi-classes
true_positive_matrix = zeros(img_index,c);
for img_index = 1: 4952
    for c = 1:num_classes
        color = all_color(mod(c, size(all_color,2))+1);
        annotation_class = training_data(img_index,c).bboxes;

        detection_class = chosen_detection{img_index, c};
        bbox_reg_class = detection_class(:, 1:4);
        probs_class = detection_class(:,5);
        chosen_detection_class = find(probs_class > chosen_threshold);
        bbox_reg_class = bbox_reg_class(1:size(chosen_detection_class,1),:);
        
        % Compute the overlap of annotation ground truth and detection
        % true positive if overlab > 50%
        if size(annotation_class,1) ~= 0 && size(chosen_detection_class,1) ~= 0
            true_positive = 0;
            for aa = 1:size(annotation_class,1)
                area = annotation_class(aa,3) * annotation_class(aa,4);
                for cc = 1:size(chosen_detection_class,1)
                    if rectint(annotation_class(aa,:), xywh_format(bbox_reg_class(cc,:))) / area >= 0.5
                        true_positive = true_positive + 1;
                    end
                end
            end
            true_positive_matrix(img_index,c) = true_positive;
        end
    end
end

true_positive_matrix = sum(true_positive_matrix,2);
[~, img_index_sort] = sort(true_positive_matrix, 'descend');
% [~, img_index_chosen] = max(true_positive_matrix);
img_index_chosen = img_index_sort(2);


% Show example contains true positive detections of multi-classes
img_part2 = imread(fullfile(images_dir, annotations{img_index_chosen}.filename));
all_color = 'rgbycm';
figure;
imshow(img_part2);
hold on
for c = 1:num_classes
    clear chosen_detection_class;
    color = all_color(mod(c, size(all_color,2))+1);
    detection_class = chosen_detection{img_index_chosen, c};
    bbox_reg_class = detection_class(:, 1:4);
    probs_class = detection_class(:,5);
    chosen_detection_class = find(probs_class > 0);

    for i = 1:size(chosen_detection_class,1)
       x = bbox_reg_class(chosen_detection_class(i),1);
       y = bbox_reg_class(chosen_detection_class(i),2);
       w = bbox_reg_class(chosen_detection_class(i),3) - x + 1;
       h = bbox_reg_class(chosen_detection_class(i),4) - y + 1;
       rectangle('Position', [x,y,w,h], 'EdgeColor', color);
       class_prob = sprintf('%s : %0.2f', class_names{1,c}, probs_class(chosen_detection_class(i)));
       text(double(x),double(y), class_prob, ...
           'BackgroundColor',color,'Color','k', ...
           'FontSize',6);
    end
end
hold off