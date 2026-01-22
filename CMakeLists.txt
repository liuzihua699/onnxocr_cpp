cmake_minimum_required(VERSION 3.14)
project(onnxocr VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 编译选项
option(USE_CUDA "Enable CUDA/GPU support" OFF)
option(USE_SYSTEM_ONNXRUNTIME "Use system installed ONNX Runtime" OFF)

# ==================== 依赖 ====================

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# ONNX Runtime
if(USE_SYSTEM_ONNXRUNTIME)
    # 使用系统安装的版本
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
else()
    # 使用项目内置的版本（默认）
    set(ONNXRUNTIME_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include")
    set(ONNXRUNTIME_LIBRARIES "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so")

    # 检查头文件（官方预编译包的结构）
    if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers not found in third_party/onnxruntime/include/")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_LIBRARIES}")
        message(FATAL_ERROR "libonnxruntime.so not found in third_party/onnxruntime/lib/!")
    endif()

    message(STATUS "Using bundled ONNX Runtime")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARIES}")

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
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib"
        INSTALL_RPATH "$ORIGIN"
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

# 复制 onnxruntime.so 到 build 目录
if(NOT USE_SYSTEM_ONNXRUNTIME)
    add_custom_command(TARGET onnxocr POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so"
            "$<TARGET_FILE_DIR:onnxocr>/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so.1.16.3"
            "$<TARGET_FILE_DIR:onnxocr>/"
            COMMENT "Copying libonnxruntime.so to build directory"
    )
endif()

# ==================== Demo ====================

add_executable(ocr_demo demo/main.cpp)
target_link_libraries(ocr_demo onnxocr ${OpenCV_LIBS})

set_target_properties(ocr_demo PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib:${CMAKE_BINARY_DIR}"
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)

# ==================== 安装 ====================

install(TARGETS onnxocr LIBRARY DESTINATION lib)
install(FILES onnxocr/onnxocr_api.h DESTINATION include/onnxocr)

# 安装 onnxruntime 库
if(NOT USE_SYSTEM_ONNXRUNTIME)
    install(FILES "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so"
            "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/libonnxruntime.so.1.16.3"
            DESTINATION lib)
endif()

# ==================== 信息 ====================

message(STATUS "")
message(STATUS "===== OnnxOCR =====")
message(STATUS "  USE_CUDA:               ${USE_CUDA}")
message(STATUS "  USE_SYSTEM_ONNXRUNTIME: ${USE_SYSTEM_ONNXRUNTIME}")
message(STATUS "  OpenCV:                 ${OpenCV_VERSION}")
message(STATUS "===================")