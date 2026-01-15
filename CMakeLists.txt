cmake_minimum_required(VERSION 3.14)
project(onnxocr VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 编译选项
option(USE_CUDA "Enable CUDA/GPU support" OFF)

# ==================== 依赖 ====================

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root directory")

if(ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
else()
    find_path(ONNXRUNTIME_INCLUDE_DIRS NAMES onnxruntime_cxx_api.h
        PATHS /usr/local/include /usr/include
        PATH_SUFFIXES onnxruntime)
    find_library(ONNXRUNTIME_LIBRARIES NAMES onnxruntime
        PATHS /usr/local/lib /usr/lib /usr/lib64)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIRS OR NOT ONNXRUNTIME_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found!")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/internal
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# ==================== Clipper ====================

add_library(clipper STATIC third_party/clipper.cpp)
set_target_properties(clipper PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ==================== 动态库 libonnxocr.so ====================

add_library(onnxocr SHARED onnxocr/onnxocr_api.cpp)

target_compile_definitions(onnxocr PRIVATE ONNXOCR_EXPORTS)

if(USE_CUDA)
    target_compile_definitions(onnxocr PRIVATE USE_CUDA)
endif()

target_link_libraries(onnxocr
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
    clipper
)

set_target_properties(onnxocr PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# ==================== 编译时复制头文件到 build/include ====================

set(OUTPUT_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include/onnxocr)
file(MAKE_DIRECTORY ${OUTPUT_INCLUDE_DIR})

add_custom_command(TARGET onnxocr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/onnxocr/onnxocr_api.h
        ${OUTPUT_INCLUDE_DIR}/onnxocr_api.h
    COMMENT "Copying header to build/include/onnxocr/"
)

# ==================== Demo ====================

add_executable(ocr_demo demo/main.cpp)
target_link_libraries(ocr_demo onnxocr ${OpenCV_LIBS})

# ==================== 安装 ====================

install(TARGETS onnxocr LIBRARY DESTINATION lib)
install(FILES onnxocr/onnxocr_api.h DESTINATION include/onnxocr)

# ==================== 信息 ====================

message(STATUS "")
message(STATUS "===== OnnxOCR =====")
message(STATUS "  USE_CUDA: ${USE_CUDA}")
message(STATUS "  OpenCV:   ${OpenCV_VERSION}")
message(STATUS "===================")
