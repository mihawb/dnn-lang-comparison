#!/bin/bash
if [ "$1" = "-s" ] || [ "$1" = "--serialize" ]; then 
	echo "Serializing PyTorch models..."
	if ! [ $(find . -name "*.pt" | wc -l) -gt 0 ]; then
		python model_serializer.py
	fi
else
	echo "Omitting PyTorch model serialization."
	echo "Use -s or --serialize flag to serialize models."
fi

if ! [ -d build ]; then
	mkdir build
fi
cd build

cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'` ..
# cmake -DCMAKE_PREFIX_PATH=/home/mihawb/dnn-lang-comparison/libtorch/libtorch-cxx11/share/cmake ..
# cmake -DCMAKE_PREFIX_PATH=/home/mihawb/dnn-lang-comparison/libtorch/libtorch-pre-cxx11/share/cmake ..
cmake --build . --config Release