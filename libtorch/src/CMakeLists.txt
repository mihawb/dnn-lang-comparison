find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB sources *.cpp)
add_library(src SHARED ${sources})

target_link_libraries(src "${TORCH_LIBRARIES}" ${OpenCV_LIBS})