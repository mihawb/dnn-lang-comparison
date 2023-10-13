find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB sources *.cpp)
add_library(src SHARED ${sources})

target_link_libraries(src "${TORCH_LIBRARIES}")