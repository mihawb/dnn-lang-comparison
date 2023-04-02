clc;        % clear console output
clear;      % clear workspace (deallocate variables)
close all;  % close all opened figures

%% Load built-in dataset and present random fraction of it

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

figure(1);
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

%% label counts and image size

labelCount = countEachLabel(imds);
imgSize = size(readimage(imds,1));

%% training and validation sets 

numTrainFiles = labelCount{1,'Count'} * 0.75;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

%% Network architecture definition

layers = [
    imageInputLayer([28 28 1]) % input layer, simple as
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...   % stochastic gradient descent with momentum
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...                  % whole set iterated through 4 times
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ... 
    'ValidationFrequency',30, ...       % how often to validate (about once every epoch)
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','cpu');      % GPU by default but defined explicitly to avoid surprises

%% Training

net = trainNetwork(imdsTrain,layers,options);

%% Validation set classification and accuracy

YPred = classify(net,imdsValidation); % ??? train test contamination
YValidation = imdsValidation.Labels;  % gonna let it slide since it's just a POC

accuracy = sum(YPred == YValidation)/numel(YValidation)