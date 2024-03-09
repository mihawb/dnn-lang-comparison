clc;        
clear;      
close all;

%% config

% in matlab the highest resolution for time measuremets is a FUCKING SECOND

filterSize = 4;
numFilters = 64; % feature map size

numLatentInputs = 100;
projectionSize = [4 4 512];

scale = 0.2;
inputSize = [64 64 3];

fhand = fopen("../results/matlab_DCGAN.csv", "w");
fprintf(fhand, "model_name,phase,epoch,loss,performance,elapsed_time\n");


%% model definitions

layersGenerator = [
    featureInputLayer(numLatentInputs)
    projectAndReshapeLayer(projectionSize)

    transposedConv2dLayer(filterSize,4*numFilters)
    batchNormalizationLayer
    reluLayer

    transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer

    transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer

    transposedConv2dLayer(filterSize,3,Stride=2,Cropping="same")
    tanhLayer
];

layersDiscriminator = [
    imageInputLayer(inputSize,Normalization="none")

    convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
    leakyReluLayer(scale)

    convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)

    convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)

    convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)

    convolution2dLayer(4,1)
    sigmoidLayer
];

% analyzeNetwork(layersDiscriminator);

netG = dlnetwork(layersGenerator);
netD = dlnetwork(layersDiscriminator);

numEpochs = 8;
miniBatchSize = 96;
learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
flipProb = 0.35;

celeba_location = "../datasets/celeba_tiny/img_align_celeba/";
N = 15000;
t_begin = tic;
celeba_ds = loadCELEBA(celeba_location, miniBatchSize, 64, N);
celeba_load_time = toc(t_begin);
fprintf(fhand, "CELEBA,read,%d,-1,-1,%f\n", i, celeba_load_time);

mbq = minibatchqueue(celeba_ds, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFormat="SSCB");

trailingAvgG = [];
trailingAvgSqG = [];
trailingAvg = [];
trailingAvgSqD = [];

numIterationsPerEpoch = floor(N/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

epoch = 0;
iteration = 0;


while epoch < numEpochs
    epoch = epoch + 1;

    % Reset and shuffle datastore.
    shuffle(mbq);

    fprintf("Epoch %d begins.\n", epoch);
    t_epoch_begin = tic;
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        X = next(mbq);

        Z = randn(numLatentInputs,miniBatchSize,"single");
        Z = dlarray(Z,"CB");
        Z = gpuArray(Z);

        % Evaluate the gradients of the loss with respect to the learnable
        % parameters, the generator state, and the network scores using
        % dlfeval and the modelLoss function.
        [~,~,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
            dlfeval(@modelLoss,netG,netD,X,Z,flipProb);
        netG.State = stateG;

        % Update the discriminator network parameters.
        [netD,trailingAvg,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
            trailingAvg, trailingAvgSqD, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the generator network parameters.
        [netG,trailingAvgG,trailingAvgSqG] = adamupdate(netG, gradientsG, ...
            trailingAvgG, trailingAvgSqG, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
    end

    t_epoch_elapsed = toc(t_epoch_begin);
    fprintf("Epoch %d processed in %f seconds.\n", epoch, t_epoch_elapsed);
    fprintf(fhand, "DCGAN,training,%d,-1,-1,%f\n", ...
        epoch,t_epoch_elapsed);
end

%% evaluation

numObservations = 1024;
ZNew = randn(numLatentInputs,numObservations,"single");
ZNew = dlarray(ZNew,"CB");
ZNew = gpuArray(ZNew);

t_eval_begin = tic;
XGeneratedNew = predict(netG,ZNew);
t_eval_elapsed = toc(t_eval_begin);
fprintf(fhand, "DCGAN,evaluation,1,-1,-1,%f\n", t_eval_elapsed);

%% latency

for i=1:numEpochs
    ZNew = randn(numLatentInputs,1,"single");
    ZNew = dlarray(ZNew,"CB");
    ZNew = gpuArray(ZNew);
    t_latency_begin = tic;
    XGeneratedNew = predict(netG,ZNew);
    t_latency_elapsed = toc(t_latency_begin);
    fprintf(fhand, "DCGAN,latency,%d,-1,-1,%f\n", i, t_latency_elapsed);
end

fclose(fhand);