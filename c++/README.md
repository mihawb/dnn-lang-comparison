# Bare cuDNN performance tests
Based on Jack Han's project available [here](https://github.com/haanjack/mnist-cudnn).  
Modifications include measuring training and inference time, support for multiple epochs training, etc.  
With author's copyright notice attached.

## How to use
```bash 
$ bash ./datasets/mnist-digits/download_mnist.sh
$ cd c++
$ make
$ ./fcnet.out fcnet
```

## Expected output
```bash
== MNIST training with CUDNN ==
[TRAIN]
loading ../datasets/mnist-digits/train-images-idx3-ubyte
loaded 60000 items..
.. model Configuration ..
CUDA: conv1
CUDA: pool
CUDA: conv2
CUDA: pool
CUDA: dense1
CUDA: relu
CUDA: dense2
CUDA: softmax
conv1: Available Algorithm Count [FWD]: 8
conv1: Available Algorithm Count [BWD-filter]: 7
conv1: Available Algorithm Count [BWD-data]: 6
.. initialized conv1 layer ..
conv2: Available Algorithm Count [FWD]: 8
conv2: Available Algorithm Count [BWD-filter]: 7
conv2: Available Algorithm Count [BWD-data]: 6
.. initialized conv2 layer ..
.. initialized dense1 layer ..
.. initialized dense2 layer ..
epoch:  1, step:  200, loss: 0.391, accuracy: 68.922%
epoch:  1, step:  400, loss: 0.247, accuracy: 89.516%
epoch:  1, step:  600, loss: 0.267, accuracy: 90.219%
epoch:  1, step:  800, loss: 0.460, accuracy: 90.312%
epoch:  1, step: 1000, loss: 0.431, accuracy: 90.094%
epoch:  1, step: 1200, loss: 0.464, accuracy: 90.406%
epoch:  1, step: 1400, loss: 0.274, accuracy: 90.094%
epoch:  1, step: 1600, loss: 0.289, accuracy: 90.188%
epoch:  1, step: 1800, loss: 0.223, accuracy: 90.219%
Epoch time: 3215.214 ms
epoch:  2, step:  200, loss: 0.302, accuracy: 90.422%
epoch:  2, step:  400, loss: 0.242, accuracy: 90.156%
epoch:  2, step:  600, loss: 0.267, accuracy: 90.188%
epoch:  2, step:  800, loss: 0.460, accuracy: 90.312%
epoch:  2, step: 1000, loss: 0.431, accuracy: 90.094%
epoch:  2, step: 1200, loss: 0.464, accuracy: 90.406%
epoch:  2, step: 1400, loss: 0.274, accuracy: 90.094%
epoch:  2, step: 1600, loss: 0.289, accuracy: 90.188%
epoch:  2, step: 1800, loss: 0.223, accuracy: 90.219%
Epoch time: 2561.446 ms
[INFERENCE]
loading ../datasets/mnist-digits/t10k-images-idx3-ubyte
loaded 10000 items..
conv1: Available Algorithm Count [FWD]: 8
conv1: Available Algorithm Count [BWD-filter]: 7
conv1: Available Algorithm Count [BWD-data]: 6
conv2: Available Algorithm Count [FWD]: 8
conv2: Available Algorithm Count [BWD-filter]: 7
conv2: Available Algorithm Count [BWD-data]: 6
loss: 0.465, accuracy: 88.000%
Done.
```
