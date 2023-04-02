clc;
clear;
close all;

[train_imgs, train_labels] = loadImgsAndLabels( ...
                            'train-images.idx3-ubyte', ...
                            'train-labels.idx1-ubyte' ...
                            );

samples = train_imgs(:,1:16);
samples = reshape(samples, 28, 28, 16);

for i = 1:16
    subplot(4,4,i)
    imshow(samples(:,:,i))
    title(train_labels(i))
end