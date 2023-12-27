#!/bin/bash

cd cudnn
make clean && make
./train.out fcnet
./train.out scvnet

cd ../python/pytorch
python clf_benchmarks.py 

cd ../tensorflow
python clf_benchmarks.py

cd ../../libtorch
./run_build.sh
./build/benchmark_runner

cd ..
matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/FullyConnectedNet_test.m'); exit;"
matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/SimpleConvNet_test.m'); exit;"
matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/MobileNet_v2_test.m'); exit;"
