if ! [ -d build ]; then
	mkdir build
fi
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release