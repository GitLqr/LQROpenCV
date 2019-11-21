cmake_minimum_required(VERSION 3.4.1)

############# OpenCV 环境 #############
# 设置 OpenCV jni 路径
set(OpenCV_DIR ../opencvSDK/sdk/native/jni)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
elseif (OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif (OpenCV_FOUND)

include_directories("${OpenCV_DIR}/include")
############# OpenCV 环境 #############

add_library(DetectionBasedTracker_jni
        SHARED
        src/main/cpp/DetectionBasedTracker_jni.cpp)

target_link_libraries(DetectionBasedTracker_jni
        ${OpenCV_LIBS}
        log)

add_library(HandTracker
        SHARED
        src/main/cpp/HandTracker.cpp)

target_link_libraries(HandTracker
        ${OpenCV_LIBS}
        log)
