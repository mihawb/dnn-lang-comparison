#!/bin/bash

run_tests() {
	cd cudnn
	make clean && make
	./train.out fcnet
	./train.out scvnet
	# ./train.out ecvnet

	cd ../python/tensorflow
	python clf_benchmarks.py 

	cd ../pytorch
	python clf_benchmarks.py

	cd ../../libtorch
	./run_build.sh -s
	./build/benchmark_runner

	cd ..
	matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/FullyConnectedNet_test.m'); exit;"
	matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/SimpleConvNet_test.m'); exit;"
	matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/MobileNet_v2_test.m'); exit;"
	matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/ResNet_50_test.m'); exit;"
	matlab -nodisplay -nosplash -nodesktop -r "run('$(pwd)/matlab/DCGAN_test.m'); exit;"
}

dest="results_ultimate"
for i in {1..15}
do
	echo "Running iteration $i of benchmarks"
	mkdir results
	run_tests
	mv results/{pytorch-*.csv,pytorch.csv}
	mv results/{tensorflow-*.csv,tensorflow.csv}
	mv results "${dest}_${i}"
done