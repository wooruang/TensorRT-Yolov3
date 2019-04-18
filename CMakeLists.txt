cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

# Add wrapper.
# If you do not install at '/usr' or '/usr/local' directories, can set a path of root of your TensorRT.
set(TENSORRT_ROOT "$ENV{HOME}/Downloads/TensorRT-5.0.2.6.Ubuntu-18.04.1.x86_64-gnu.cuda-10.0.cudnn7.3/TensorRT-5.0.2.6")

set(TRT_WRAPPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tensorRTWrapper/code)
add_subdirectory(${TRT_WRAPPER_DIR})
include_directories(${TRT_WRAPPER_DIR}/include)

# Add opencv.
# Set a build path of custom of OpenCV.
#set(OpenCV_ROOT "")
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

# This project dir.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#build runYolov3
add_executable(runYolov3 main.cpp eval.cpp dataReader.cpp src/utils.cpp src/Yolov3TensorRT.cpp)
target_link_libraries(runYolov3 TrtNet ${OpenCV_LIBS})

install(TARGETS runYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
