function rb = resBlock(in_channels, out_channels)
rb = [
    convolution2dLayer(3, in_channels, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer  

    convolution2dLayer(3, out_channels, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer  

    maxPooling2dLayer(2, 'Stride', 2)
];
end

