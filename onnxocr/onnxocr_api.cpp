/**
 * OnnxOCR C API 实现
 */

#include "onnxocr/onnxocr_api.h"
#include "onnxocr.hpp"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#include <mutex>

// 全局错误信息
static thread_local std::string g_last_error;

// 内部OCR包装类
struct OcrInstance {
    std::unique_ptr<onnxocr::TextSystem> ocr;
    onnxocr::Config config;
};

extern "C" {

ONNXOCR_API OcrHandle ocr_create(const OcrConfig* config) {
    if (!config) {
        g_last_error = "Config is null";
        return nullptr;
    }
    
    try {
        auto* instance = new OcrInstance();
        
        // 设置配置
        if (config->det_model_path) {
            instance->config.det_model_path = config->det_model_path;
        }
        if (config->rec_model_path) {
            instance->config.rec_model_path = config->rec_model_path;
        }
        if (config->cls_model_path) {
            instance->config.cls_model_path = config->cls_model_path;
        }
        if (config->rec_char_dict_path) {
            instance->config.rec_char_dict_path = config->rec_char_dict_path;
        }
        
        instance->config.use_gpu = config->use_gpu != 0;
        instance->config.gpu_id = config->gpu_id;
        instance->config.use_angle_cls = config->use_angle_cls != 0;
        
        if (config->drop_score > 0) {
            instance->config.drop_score = config->drop_score;
        }
        
        // 创建OCR实例
        instance->ocr = std::make_unique<onnxocr::TextSystem>(instance->config);
        
        return static_cast<OcrHandle>(instance);
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

ONNXOCR_API void ocr_destroy(OcrHandle handle) {
    if (handle) {
        auto* instance = static_cast<OcrInstance*>(handle);
        delete instance;
    }
}

// 内部函数：将OCR结果转换为C结构
static void convert_result(const onnxocr::OCRResult& ocr_result, OcrResult* result) {
    result->count = static_cast<int>(ocr_result.size());
    
    if (result->count == 0) {
        result->lines = nullptr;
        return;
    }
    
    result->lines = new OcrTextLine[result->count];
    
    for (int i = 0; i < result->count; ++i) {
        const auto& line = ocr_result[i];
        
        // 复制坐标
        for (int j = 0; j < 4; ++j) {
            result->lines[i].box[j * 2] = line.box[j].x;
            result->lines[i].box[j * 2 + 1] = line.box[j].y;
        }
        
        // 复制文本
        result->lines[i].text = new char[line.text.size() + 1];
        std::strcpy(result->lines[i].text, line.text.c_str());
        
        // 复制置信度
        result->lines[i].score = line.score;
    }
}

ONNXOCR_API int ocr_recognize_file(OcrHandle handle, const char* image_path, OcrResult* result) {
    if (!handle || !image_path || !result) {
        g_last_error = "Invalid parameters";
        return -1;
    }
    
    try {
        auto* instance = static_cast<OcrInstance*>(handle);
        
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            g_last_error = "Failed to load image: " + std::string(image_path);
            return -2;
        }
        
        auto ocr_result = (*instance->ocr)(img);
        convert_result(ocr_result, result);
        
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -3;
    }
}

ONNXOCR_API int ocr_recognize_buffer(OcrHandle handle, const unsigned char* image_data,
                                      int width, int height, int channels, OcrResult* result) {
    if (!handle || !image_data || !result) {
        g_last_error = "Invalid parameters";
        return -1;
    }
    
    if (channels != 3) {
        g_last_error = "Only 3-channel BGR images are supported";
        return -2;
    }
    
    try {
        auto* instance = static_cast<OcrInstance*>(handle);
        
        cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(image_data));
        
        auto ocr_result = (*instance->ocr)(img);
        convert_result(ocr_result, result);
        
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -3;
    }
}

ONNXOCR_API void ocr_free_result(OcrResult* result) {
    if (result && result->lines) {
        for (int i = 0; i < result->count; ++i) {
            delete[] result->lines[i].text;
        }
        delete[] result->lines;
        result->lines = nullptr;
        result->count = 0;
    }
}

ONNXOCR_API const char* ocr_get_last_error() {
    return g_last_error.c_str();
}

} // extern "C"
