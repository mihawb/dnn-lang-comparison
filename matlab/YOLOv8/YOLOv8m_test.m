clc;        
clear;      
close all;

fhand = fopen("../../results_ultimate_0/matlab_YOLOv8.csv", "w");
fprintf(fhand, "framework,model_name,phase,epoch,loss,performance,elapsed_time\n");

imgs = dir(fullfile('../../datasets/coco2017_val', '*.jpg'));
net = yolov8ObjectDetector('yolov8m');

for i=1:5000
    img = imread(fullfile(imgs(i).folder, imgs(i).name));
    if length(size(img)) ~= 3
        img = cat(3, img, img, img);
    end
    g_img = gpuArray(img);

    t_begin = tic;
    [bboxes, scores, labels] = detect(net, g_img);
    t_elapsed = toc(t_begin);

    fprintf(fhand, "Matlab,YOLOv8m,latency,%d,-1,-1,%f\n", i, t_elapsed);
    fprintf("Image %d: %fs\n", i, t_elapsed);
end

fclose(fhand);