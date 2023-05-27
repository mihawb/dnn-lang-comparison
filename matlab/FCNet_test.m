clc;        
clear;      
close all;

%% loading data

[x_train, y_train] = loadMNISTImgsAndLabels( ...
    '../datasets/mnist-digits/train-images.idx3-ubyte', ...
    '../datasets/mnist-digits/train-labels.idx1-ubyte' ...
);

[x_test, y_test] = loadMNISTImgsAndLabels( ...
    '../datasets/mnist-digits/t10k-images.idx3-ubyte', ...
    '../datasets/mnist-digits/t10k-labels.idx1-ubyte' ...
);
test_ds = {x_test, y_test};

global training_state
training_state = [];

%% config

momentum = 0.9;
lr = 0.01;
epochs = 5;
batch_size = 64;
val_freq = floor(size(x_train, 4) / batch_size);

options = trainingOptions(          ...
	"sgdm", 						...
	Momentum = 0.9,					...
	InitialLearnRate = 0.01, 		...
	MaxEpochs = 5, 					...
	MiniBatchSize = 64, 			...
	Shuffle = "every-epoch", 		...
    ValidationData = test_ds,       ... 
    ValidationFrequency = val_freq, ... 
	ExecutionEnvironment = "cpu",   ...
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