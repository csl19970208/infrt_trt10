cmake_minimum_required(VERSION 3.0.2)
project(infer_trt10 VERSION 5.0.0)

# 设置 C++17 标准
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/cuda/lib)

include_directories(/usr/src/TensorRT-10.7.0.23/include)
link_directories(/usr/src/TensorRT-10.7.0.23/lib)

#include_directories(${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(
  include
  # ${catkin_INCLUDE_DIRS}
)

file(GLOB SOURCES "src/infer_trt10.cpp")

cuda_add_executable(infer_trt10 ${SOURCES})
#cuda_add_library(infer_trt10 SHARED src/infer_trt10.cpp)
target_link_libraries(infer_trt10
        ${TENSORRT_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        -lpthread cuda hiredis
        nvinfer
        cudart
        nvonnxparser
        )
add_definitions(-O2 -pthread)
