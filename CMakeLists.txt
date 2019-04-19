cmake_minimum_required(VERSION 3.0)

set(LIB_MAJOR_VERSION "0")
set(LIB_MINOR_VERSION "0")
set(LIB_PATCH_VERSION "0")
set(LIB_VERSION_STRING "${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${LIB_PATCH_VERSION}")

if (CMAKE_VERSION VERSION_LESS 3.0)
    PROJECT(TensorRTYolov3 CXX)
    set(PROJECT_VERSION_MAJOR "${LIB_MAJOR_VERSION}")
    set(PROJECT_VERSION_MINOR "${LIB_MINOR_VERSION}")
    set(PROJECT_VERSION_PATCH"${LIB_PATCH_VERSION}")
    set(PROJECT_VERSION "${LIB_VERSION_STRING}")
else()
    cmake_policy(SET CMP0048 NEW)
    PROJECT(TensorRTYolov3 VERSION "${LIB_VERSION_STRING}" LANGUAGES CXX)
endif()

message("${PROJECT_VERSION} ${PROJECT_VERSION_MAJOR} ${PROJECT_VERSION_MINOR} ${PROJECT_VERSION_PATCH}" )

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -fPIC")

# Add wrapper.
# If you do not install at '/usr' or '/usr/local' directories, can set a path of root of your TensorRT.
set(TENSORRT_ROOT "")
set(USE_TENSORRT_SHARED ON)
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

# Add TensorRTYolov3 library.
add_library(TensorRTYolov3 SHARED eval.cpp dataReader.cpp src/utils.cpp src/Yolov3TensorRT.cpp)
set_target_properties(TensorRTYolov3 PROPERTIES VERSION ${PROJECT_VERSION})
target_link_libraries(TensorRTYolov3 TrtNet ${OpenCV_LIBS})

# Add runYolov3 executable.
add_executable(runYolov3 yolov3tensorrt_main.cpp)
target_link_libraries(runYolov3 TensorRTYolov3)

install(TARGETS runYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
install(TARGETS TensorRTYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
