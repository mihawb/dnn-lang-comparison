clc;
clear;
close all;

%% MNIST
[mnist_imgs, mnist_labels] = loadMNISTImgsAndLabels( ...
    '../datasets/mnist-digits/train-images-idx3-ubyte', ...
    '../datasets/mnist-digits/train-labels-idx1-ubyte' ...
);


figure(1);
for i = 1:16
    subplot(4,4,i)
    imshow(mnist_imgs(:,:,i))
    title(mnist_labels(i))
end

%% CIFAR-10
cifar_location = "../datasets/cifar-10-matlab";
downloadCIFARData(cifar_location);
[cifar_imgs, cifar_labels, x_irrelevant, y_irrelevant] = loadCIFARData(cifar_location);

% native res
figure(2);
for i = 1:16
    subplot(4,4,i)
    imshow(cifar_imgs(:,:,:,i))
    title(cifar_labels(i))
end

% upscaled to 224x224
figure(3);
for i = 17:20
    subplot(2,2,i-16)
    img = imresize(cifar_imgs(:,:,:,i), [224 NaN]);
    imshow(img)
    title(cifar_labels(i))
end

%% CELEB-A
celeba_location = "../datasets/celeba_tiny/img_align_celeba/";
celeba_imgs = loadCELEBA(celeba_location, 1, 64, 16);
figure(4);

for i = 1:16
    subplot(4,4,i);
    img = cell2mat(read(celeba_imgs));
    imshow(img);
end

%% CELEBA-A Tiny loading benchmark
N = 3000; % can load 30k on its own but not alongside other datasets
t_begin = tic;
celeba_ds = loadCELEBA(celeba_location, 96, 64, N);
t_elapsed = toc(t_begin);
fprintf("Loading %d CELEB-A images took %f seconds.\n", N, t_elapsed);
