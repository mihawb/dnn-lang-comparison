clc;        
clear;      
close all;

%% loading data

adamroot = "../datasets/ADAM/Training1200/";
[x_train, y_train, x_test, y_test] = loadADAM(adamroot, 0.8);

train_ds = {x_train, y_train};
test_ds = {x_test, y_test};

global training_state
training_state = [];

%% config

momentum = 0.9;
lr = 0.01;
epochs = 8;
batch_size = 96;
val_freq = floor(size(x_train, 4) / batch_size);
verbose_freq = floor(val_freq / 4);
% verbose output requires inference which severly slows down learning process

options = trainingOptions(          ...
	"sgdm", 						...
	Momentum = momentum,			...
	InitialLearnRate = lr, 		    ...
	MaxEpochs = epochs,             ...
	MiniBatchSize = batch_size,     ...
	Shuffle = "every-epoch", 		...
    ValidationData = test_ds,       ...
    ValidationFrequency = val_freq, ...
    VerboseFrequency = verbose_freq,... 
	ExecutionEnvironment = "gpu",   ...
	Plots = "none",					...
    OutputFcn=@saveTrainingState    ...
);

%% net definition

chans = 16;

layers = [
    imageInputLayer([256 256 3])
    
    resBlock(3, chans)
    resBlock(chans, 2*chans)
    resBlock(2*chans, 4*chans)
    resBlock(4*chans, 8*chans)

    convolution2dLayer(3, 16*chans)
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(2)
];

%% Training

analyzeNetwork(layers);

net = trainNetwork(x_train, y_train, layers, options);

results = struct2table(training_state);
writetable(results, "../results/matlab_FullyConnectedNet.csv");

%% testing accuracy

inference_time = timeit(@() classify(net, x_test, MiniBatchSize=batch_size*2));
outputs = classify(net, x_test, MiniBatchSize=batch_size*2);
test_acc = mean(outputs == y_test);

fhand = fopen("../results/matlab_SingleObjectDetectionNet.csv", "a+");
fprintf(fhand, "1,1,%f,,,,,,%f,,inference\n", inference_time,test_acc);

%% latency

for i=1:epochs
    img = x_test(:,:,:,i);
    t_latency_begin = tic;
    cls = predict(net,img);
    t_latency_elapsed = toc(t_latency_begin);
    fprintf(fhand, "%d,1,%f,,,,,,,,latency\n", i, t_latency_elapsed);
end

fclose(fhand);