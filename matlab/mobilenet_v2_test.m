clc;        
clear;      
close all;

%% config
momentum = 0.9;
lr = 0.01;
epochs = 8;
batch_size = 96;

%% loading data

cifar_location = "../datasets/cifar-10-matlab";
downloadCIFARData(cifar_location);
[x_train, y_train, x_test, y_test] = loadCIFARData(cifar_location);

% some more config
val_freq = floor(size(x_train, 4) / batch_size);
verbose_freq = floor(val_freq / 4);
% verbose output requires inference which severly slows down learning process


% train_ds = augmentedImageDatastore([32 32 3], x_train, y_train);
% test_ds = augmentedImageDatastore([32 32 3], x_test, y_test);
train_ds = {x_train, y_train};
test_ds = {x_test, y_test};

global training_state
training_state = [];

%% options

options = trainingOptions(              ...
	"sgdm", 						    ...
	Momentum = momentum,			    ...
	InitialLearnRate = lr, 		        ...
	MaxEpochs = epochs,                 ...
	MiniBatchSize = batch_size,         ...
	Shuffle = "every-epoch", 		    ...
    ValidationData = test_ds,           ...
    ValidationFrequency = val_freq,     ...
    VerboseFrequency = verbose_freq,    ...
	ExecutionEnvironment = "gpu",       ...
	Plots = "none",					    ...
    OutputFcn=@saveTrainingState        ...
);

%% network definition

new_inputs = imageInputLayer([32 32 3], ...
    Name = "new_input_1", ...
    Normalization = "zscore" ...
);

new_logits = fullyConnectedLayer(10, ...
    Name = "NewLogits", ...
    BiasLearnRateFactor = 10, ...
    WeightLearnRateFactor = 10 ...
);

model = mobilenetv2(Weights="none");
model = replaceLayer(model, "Logits", new_logits);
model = replaceLayer(model, "input_1", new_inputs);

%% training

net = trainNetwork(x_train, y_train, model, options);

results = struct2table(training_state);
writetable(results, "../results/matlab_MobileNet-v2.csv");

%% testing accuracy

inference_time = timeit(@() classify(net, x_test, MiniBatchSize=batch_size*2));
outputs = classify(net, x_test, MiniBatchSize=batch_size*2);
test_acc = mean(outputs == y_test);

fhand = fopen("../results/matlab_MobileNet-v2.csv", "a+");
fprintf(fhand, "1,1,%f,,,,,,%f,,inference", inference_time,test_acc);
fclose(fhand);