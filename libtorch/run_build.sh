#!/bin/bash
if ! [ $(find . -name "*.pt" | wc -l) -gt 0 ]; then
	python model_serializer.py
fi

if ! [ -d build ]; then
	mkdir build
fi
cd build

cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'` ..
# cmake -DCMAKE_PREFIX_PATH=/home/mihawb/dnn-lang-comparison/libtorch/libtorch-cxx11/share/cmake ..
# cmake -DCMAKE_PREFIX_PATH=/home/mihawb/dnn-lang-comparison/libtorch/libtorch-pre-cxx11/share/cmake ..
cmake --build . --config Release