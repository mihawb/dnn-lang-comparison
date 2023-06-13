#!/bin/bash
./datasets/mnist-digits/download_mnist.sh
cd c++ 
make clean && make
./train.out fcnet
./train.out scvnet
cd ../python
python torch_benchmarks.py 
python tf_benchmarks.py
cd ..
matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/FCNet_test.m'); exit;"
matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/mobilenet_v2_test.m'); exit;"
