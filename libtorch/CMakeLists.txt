cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(benchmark_runner)
set(CAFFE2_USE_CUDNN ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(benchmark_runner benchmark_runner.cpp)
target_link_libraries(benchmark_runner "${TORCH_LIBRARIES}")

add_subdirectory(src)
target_link_libraries(benchmark_runner src)
target_include_directories(benchmark_runner PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/src"
                          )

set_property(TARGET benchmark_runner PROPERTY CXX_STANDARD 17)