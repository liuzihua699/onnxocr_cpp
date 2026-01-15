/**
 * 调用 libonnxocr.so 动态库的示例
 */

#include <iostream>
#include <cstring>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "onnxocr/onnxocr_api.h"
#include "chrono"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [options]\n"
                  << "Options:\n"
                  << "  --det_model <path>   Detection model\n"
                  << "  --rec_model <path>   Recognition model\n"
                  << "  --dict <path>        Dictionary file\n"
                  << "  --use_gpu            Use GPU\n";
        return 1;
    }
    
    // 默认配置
    OcrConfig config;
    std::memset(&config, 0, sizeof(config));
    config.det_model_path = "models/det.onnx";
    config.rec_model_path = "models/rec.onnx";
    config.rec_char_dict_path = "models/ppocr_keys.txt";
    config.use_gpu = 0;
    config.gpu_id = 0;
    config.use_angle_cls = 0;
    config.drop_score = 0.5f;
    
    const char* image_path = argv[1];
    
    // 解析参数
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--det_model") == 0 && i + 1 < argc) {
            config.det_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--rec_model") == 0 && i + 1 < argc) {
            config.rec_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--dict") == 0 && i + 1 < argc) {
            config.rec_char_dict_path = argv[++i];
        } else if (std::strcmp(argv[i], "--cls_model") == 0 && i + 1 < argc) {
            config.cls_model_path = argv[++i];
            config.use_angle_cls = 1;
        } else if (std::strcmp(argv[i], "--use_gpu") == 0) {
            config.use_gpu = 1;
        }
    }
    
    std::cout << "Initializing OCR...\n";
    std::cout << "  GPU: " << (config.use_gpu ? "ON" : "OFF") << "\n";
    
    // 创建OCR实例
    OcrHandle handle = ocr_create(&config);
    if (!handle) {
        std::cerr << "Failed to create OCR: " << ocr_get_last_error() << "\n";
        return 1;
    }
    
    std::cout << "Processing: " << image_path << "\n";
    
    // 识别
    OcrResult result;
    cv::Mat img = cv::imread(image_path);

    auto start = std::chrono::high_resolution_clock::now();
//    int ret = ocr_recognize_file(handle, image_path, &result);
    int ret = ocr_recognize_buffer(handle, img.data, img.cols, img.rows, img.channels(), &result);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "ocr_recognize_buffer cost=" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    
    if (ret != 0) {
        std::cerr << "OCR failed: " << ocr_get_last_error() << "\n";
        ocr_destroy(handle);
        return 1;
    }
    
    // 输出结果
    std::cout << "\n===== Results (" << result.count << " lines) =====\n";
    for (int i = 0; i < result.count; ++i) {
        std::cout << "[" << i + 1 << "] " << result.lines[i].text 
                  << " (score: " << result.lines[i].score << ")\n";
    }
    
    // 释放
    ocr_free_result(&result);
    ocr_destroy(handle);
    
    return 0;
}
