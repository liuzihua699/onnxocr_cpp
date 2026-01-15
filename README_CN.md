## OnnxOCR C++

> 引用自 [OnnxOCR](https://github.com/RapidAI/OnnxOCR)


## 环境要求

- C++17 或更高
- OpenCV 4.x
- ONNX Runtime 1.11.0+

## CentOS 7 环境准备

## 安装 ONNX Runtime

```
# 基本环境编译
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
# ./build.sh --config Release --build_shared_lib --parallel
./build.sh --config Release --build_shared_lib --parallel --skip_submodule_sync --allow_running_as_root --cmake_path /home/lzh/cmake3.8/cmake-3.28.0-linux-x86_64/bin/cmake
cd build/Linux/Release
sudo make install
```



## 编译

```bash
cd cpp
mkdir build && cd build

# 基本编译
cmake ..
make -j$(nproc)

# GPU 支持
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
sudo make install
```

## 运行

```bash
# 准备模型文件
mkdir -p models
# 将 det.onnx, rec.onnx, ppocr_keys.txt 复制到 models 目录

# 运行
./ocr_demo test.jpg --det_model ../models/det.onnx --rec_model ../models/rec.onnx --dict ../models/ppocr_keys.txt
```

![demo.png](demo.png)


## 代码结构

```
cpp/
├── include/
│   └── onnxocr/
│       ├── onnxocr.hpp        # 主入口 (include this)
│       ├── types.hpp          # 类型定义
│       ├── config.hpp         # 配置
│       ├── utils.hpp          # 工具函数
│       ├── preprocess.hpp     # 预处理
│       ├── db_postprocess.hpp # 检测后处理
│       ├── ctc_postprocess.hpp# 识别后处理
│       ├── cls_postprocess.hpp# 分类后处理
│       ├── onnx_session.hpp   # ONNX会话管理
│       ├── detector.hpp       # 文本检测器
│       ├── recognizer.hpp     # 文本识别器
│       ├── classifier.hpp     # 文本分类器
│       └── text_system.hpp    # OCR系统
├── third_party/
│   ├── clipper.hpp            # Clipper 多边形库
│   └── clipper.cpp
├── src/
│   └── main.cpp               # 示例程序
└── CMakeLists.txt
```

## API 使用示例

### CMake引用

1. 有安装
```cmake
find_package(onnxocr REQUIRED)
target_link_libraries(your_app onnxocr)
```

头文件引入：
```cpp
#include <onnxocr/onnxocr_api.h>
```

2. 未安装
```cmake
find_package(onnxruntime REQUIRED)
set(ONNXOCR_DIR "/path/to/OnnxOCR/cpp")
target_include_directories(your_app PRIVATE
        ${ONNXOCR_DIR}/onnxocr
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(your_app
        ${ONNXOCR_DIR}/build/libonnxocr.so
        ${OpenCV_LIBS}
        onnxruntime
)
```



### 基本使用

```cpp
#include "onnxocr/onnxocr.hpp"

// 配置
onnxocr::Config config;
config.det_model_path = "models/det.onnx";
config.rec_model_path = "models/rec.onnx";
config.rec_char_dict_path = "models/ppocr_keys.txt";
config.use_gpu = false;

// 初始化
onnxocr::OCR ocr(config);

// 识别
cv::Mat img = cv::imread("test.jpg");
auto results = ocr(img);

// 输出结果
for (const auto& line : results) {
    std::cout << line.text << " [" << line.score << "]" << std::endl;
}
```

### 仅检测

```cpp
auto boxes = ocr.detect(img);
for (const auto& box : boxes) {
    // box[0], box[1], box[2], box[3] 为四个角点
}
```

### 仅识别

```cpp
std::vector<cv::Mat> crops = {...};  // 裁剪好的文本图像
auto rec_results = ocr.recognize(crops);
```

### 启用方向分类

```cpp
config.use_angle_cls = true;
config.cls_model_path = "models/cls.onnx";
```

## 模型转换

将 PaddleOCR 模型转换为 ONNX 格式：

```bash
# 使用 paddle2onnx
paddlex --install paddle2onnx
            
# 转换检测模型
paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_det_infer --onnx_model_dir ./det_onnx --opset_version 11

# 转换识别模型
paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_rec_infer --onnx_model_dir ./rec_onnx --opset_version 11
```

## 注意事项

1. **字典文件**(ppocr_keys.txt): 必须与训练时使用的字典一致
2. **图像格式**: 输入为 BGR 格式的 cv::Mat
3. **内存**: 大批量处理时注意内存占用
4. **性能**: GPU 版本需要安装对应 CUDA 版本的 ONNX Runtime
