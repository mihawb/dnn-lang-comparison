clc;        
clear;      
close all;

%% loading data

[x_train, y_train] = loadMNISTImgsAndLabels( ...
    '../datasets/mnist-digits/train-images-idx3-ubyte', ...
    '../datasets/mnist-digits/train-labels-idx1-ubyte' ...
);

[x_test, y_test] = loadMNISTImgsAndLabels( ...
    '../datasets/mnist-digits/t10k-images-idx3-ubyte', ...
    '../datasets/mnist-digits/t10k-labels-idx1-ubyte' ...
);
test_ds = {x_test, y_test};

global training_state
training_state = [];

%% config

momentum = 0.9;
lr = 0.01;
epochs = 15;
batch_size = 32;
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

layers = [
    imageInputLayer([28 28 1])
    
    fullyConnectedLayer(800)
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

%% Training

net = trainNetwork(x_train, y_train, layers, options);

results = struct2table(training_state);
writetable(results, "../results/matlab-fcnet.csv");