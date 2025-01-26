clc;        
clear;      
close all;

%% config
momentum = 0.9;
lr = 0.01;
epochs = 12;
latency_warmup_steps = 1000;
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

fc10 = fullyConnectedLayer(10, ...
    Name = "NewLogits", ...
    BiasLearnRateFactor = 10, ...
    WeightLearnRateFactor = 10 ...
);

model = resnet50(Weights="none");
model = replaceLayer(model, "fc1000", fc10);
model = replaceLayer(model, "input_1", new_inputs);

%% training

net = trainNetwork(x_train, y_train, model, options);

results = struct2table(training_state);
writetable(results, "../results/matlab_ResNet-50.csv");

%% testing accuracy

inference_time = timeit(@() classify(net, x_test, MiniBatchSize=batch_size*2));
outputs = classify(net, x_test, MiniBatchSize=batch_size*2);
test_acc = mean(outputs == y_test);

fhand = fopen("../results/matlab_ResNet-50.csv", "a+");
fprintf(fhand, "1,1,%f,,,,,,%f,,inference\n", inference_time,test_acc);

%% latency

for i=1:epochs+latency_warmup_steps
    img = x_test(:,:,:,i);
    t_latency_begin = tic;
    cls = predict(net,img);
    t_latency_elapsed = toc(t_latency_begin);
    if i > latency_warmup_steps
        fprintf(fhand, "%d,1,%f,,,,,,-1,,latency\n", i, t_latency_elapsed);
    end
end

fclose(fhand);