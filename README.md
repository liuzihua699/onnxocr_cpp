## OnnxOCR C++

> Based on [OnnxOCR](https://github.com/RapidAI/OnnxOCR)


## Requirements

- C++17 or higher
- OpenCV 4.x
- ONNX Runtime 1.11.0+

## CentOS 7 Environment Setup

## Installing ONNX Runtime

```bash
# Build from source
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
# ./build.sh --config Release --build_shared_lib --parallel
./build.sh --config Release --build_shared_lib --parallel --skip_submodule_sync --allow_running_as_root --cmake_path /home/lzh/cmake3.8/cmake-3.28.0-linux-x86_64/bin/cmake
cd build/Linux/Release
sudo make install
```

## Build

```bash
cd cpp
mkdir build && cd build

# Basic build
cmake ..
make -j$(nproc)

# With GPU support
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
sudo make install
```

## Usage

```bash
# Prepare model files
mkdir -p models
# Copy det.onnx, rec.onnx, ppocr_keys.txt to models directory

# Run demo
./ocr_demo test.jpg --det_model ../models/det.onnx --rec_model ../models/rec.onnx --dict ../models/ppocr_keys.txt
```
![demo.png](demo.png)

## Project Structure

```
cpp/
├── include/
│   └── onnxocr/
│       ├── onnxocr.hpp        # Main entry point (include this)
│       ├── types.hpp          # Type definitions
│       ├── config.hpp         # Configuration
│       ├── utils.hpp          # Utility functions
│       ├── preprocess.hpp     # Preprocessing
│       ├── db_postprocess.hpp # Detection postprocessing
│       ├── ctc_postprocess.hpp# Recognition postprocessing
│       ├── cls_postprocess.hpp# Classification postprocessing
│       ├── onnx_session.hpp   # ONNX session management
│       ├── detector.hpp       # Text detector
│       ├── recognizer.hpp     # Text recognizer
│       ├── classifier.hpp     # Text classifier
│       └── text_system.hpp    # OCR system
├── third_party/
│   ├── clipper.hpp            # Clipper polygon library
│   └── clipper.cpp
├── src/
│   └── main.cpp               # Example program
└── CMakeLists.txt
```

## API Usage Examples

### CMake Integration

1. With installation
```cmake
find_package(onnxocr REQUIRED)
target_link_libraries(your_app onnxocr)
```

Header include:
```cpp
#include <onnxocr/onnxocr_api.h>
```

2. Without installation
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

### Basic Usage

```cpp
#include "onnxocr/onnxocr.hpp"

// Configuration
onnxocr::Config config;
config.det_model_path = "models/det.onnx";
config.rec_model_path = "models/rec.onnx";
config.rec_char_dict_path = "models/ppocr_keys.txt";
config.use_gpu = false;

// Initialize
onnxocr::OCR ocr(config);

// Recognize
cv::Mat img = cv::imread("test.jpg");
auto results = ocr(img);

// Print results
for (const auto& line : results) {
    std::cout << line.text << " [" << line.score << "]" << std::endl;
}
```

### Detection Only

```cpp
auto boxes = ocr.detect(img);
for (const auto& box : boxes) {
    // box[0], box[1], box[2], box[3] are the four corner points
}
```

### Recognition Only

```cpp
std::vector<cv::Mat> crops = {...};  // Cropped text images
auto rec_results = ocr.recognize(crops);
```

### Enable Angle Classification

```cpp
config.use_angle_cls = true;
config.cls_model_path = "models/cls.onnx";
```

## Model Conversion

Convert PaddleOCR models to ONNX format:

```bash
# Using paddle2onnx
paddlex --install paddle2onnx
            
# Convert detection model
paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_det_infer --onnx_model_dir ./det_onnx --opset_version 11

# Convert recognition model
paddlex --paddle2onnx --paddle_model_dir ./PP-OCRv5_server_rec_infer --onnx_model_dir ./rec_onnx --opset_version 11
```

## Notes

1. **Dictionary file** (ppocr_keys.txt): Must match the dictionary used during training
2. **Image format**: Input should be BGR format cv::Mat
3. **Memory**: Pay attention to memory usage when processing large batches
4. **Performance**: GPU version requires ONNX Runtime built with corresponding CUDA version
