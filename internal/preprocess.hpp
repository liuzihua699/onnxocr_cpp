#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

namespace onnxocr {
namespace preprocess {

// 检测预处理结果
struct DetPreprocessResult {
    std::vector<float> data;
    std::vector<int64_t> shape;
    float ratio_h;
    float ratio_w;
    int src_h;
    int src_w;
};

// 检测预处理
inline DetPreprocessResult det_preprocess(const cv::Mat& img, 
                                          int limit_side_len,
                                          const std::string& limit_type) {
    DetPreprocessResult result;
    result.src_h = img.rows;
    result.src_w = img.cols;
    
    int h = img.rows, w = img.cols;
    float ratio = 1.0f;
    
    // 计算缩放比例
    if (limit_type == "max") {
        if (std::max(h, w) > limit_side_len) {
            ratio = static_cast<float>(limit_side_len) / std::max(h, w);
        }
    } else if (limit_type == "min") {
        if (std::min(h, w) < limit_side_len) {
            ratio = static_cast<float>(limit_side_len) / std::min(h, w);
        }
    } else {
        ratio = static_cast<float>(limit_side_len) / std::max(h, w);
    }
    
    int resize_h = static_cast<int>(h * ratio);
    int resize_w = static_cast<int>(w * ratio);
    
    // 对齐到32的倍数
    resize_h = std::max(static_cast<int>(std::round(resize_h / 32.0) * 32), 32);
    resize_w = std::max(static_cast<int>(std::round(resize_w / 32.0) * 32), 32);
    
    result.ratio_h = static_cast<float>(resize_h) / h;
    result.ratio_w = static_cast<float>(resize_w) / w;
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    
    // 转换为float并归一化
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    // BGR -> RGB 并归一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    float mean[] = {0.406f, 0.456f, 0.485f};  // BGR顺序
    float std_val[] = {0.225f, 0.224f, 0.229f};
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std_val[i];
    }
    
    // HWC -> CHW (RGB顺序)
    result.shape = {1, 3, static_cast<int64_t>(resize_h), static_cast<int64_t>(resize_w)};
    result.data.resize(3 * resize_h * resize_w);
    
    // BGR转RGB: channels[2]->R, channels[1]->G, channels[0]->B
    for (int c = 0; c < 3; ++c) {
        int src_c = 2 - c;  // BGR to RGB
        for (int i = 0; i < resize_h; ++i) {
            const float* row = channels[src_c].ptr<float>(i);
            for (int j = 0; j < resize_w; ++j) {
                result.data[c * resize_h * resize_w + i * resize_w + j] = row[j];
            }
        }
    }
    
    return result;
}

// 识别预处理
inline std::pair<std::vector<float>, std::vector<int64_t>> 
rec_preprocess(const cv::Mat& img, int imgC, int imgH, int imgW, float max_wh_ratio) {
    int target_w = static_cast<int>(imgH * max_wh_ratio);
    target_w = std::max(target_w, 1);
    
    float h = static_cast<float>(img.rows);
    float w = static_cast<float>(img.cols);
    float ratio = w / h;
    int resized_w;
    
    if (std::ceil(imgH * ratio) > target_w) {
        resized_w = target_w;
    } else {
        resized_w = static_cast<int>(std::ceil(imgH * ratio));
    }
    resized_w = std::max(resized_w, 1);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, imgH));
    
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3);
    
    // 归一化: /255, -0.5, /0.5
    float_img /= 255.0f;
    float_img -= 0.5f;
    float_img /= 0.5f;
    
    // Padding
    cv::Mat padded = cv::Mat::zeros(imgH, target_w, CV_32FC3);
    float_img.copyTo(padded(cv::Rect(0, 0, resized_w, imgH)));
    
    // HWC -> CHW
    std::vector<float> data(imgC * imgH * target_w);
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < imgH; ++i) {
            const float* row = channels[c].ptr<float>(i);
            for (int j = 0; j < target_w; ++j) {
                data[c * imgH * target_w + i * target_w + j] = row[j];
            }
        }
    }
    
    return {data, {1, imgC, imgH, static_cast<int64_t>(target_w)}};
}

// 分类预处理
inline std::pair<std::vector<float>, std::vector<int64_t>> 
cls_preprocess(const cv::Mat& img, int imgC, int imgH, int imgW) {
    float h = static_cast<float>(img.rows);
    float w = static_cast<float>(img.cols);
    float ratio = w / h;
    int resized_w = std::min(imgW, static_cast<int>(std::ceil(imgH * ratio)));
    resized_w = std::max(resized_w, 1);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, imgH));
    
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3);
    
    float_img /= 255.0f;
    float_img -= 0.5f;
    float_img /= 0.5f;
    
    cv::Mat padded = cv::Mat::zeros(imgH, imgW, CV_32FC3);
    float_img.copyTo(padded(cv::Rect(0, 0, resized_w, imgH)));
    
    std::vector<float> data(imgC * imgH * imgW);
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < imgH; ++i) {
            const float* row = channels[c].ptr<float>(i);
            for (int j = 0; j < imgW; ++j) {
                data[c * imgH * imgW + i * imgW + j] = row[j];
            }
        }
    }
    
    return {data, {1, imgC, imgH, imgW}};
}

// 批量识别预处理
inline std::pair<std::vector<float>, std::vector<int64_t>>
rec_preprocess_batch(const std::vector<cv::Mat>& imgs, int imgC, int imgH, int target_w) {
    int batch_size = static_cast<int>(imgs.size());
    std::vector<float> batch_data(batch_size * imgC * imgH * target_w, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        const cv::Mat& img = imgs[b];
        float h = static_cast<float>(img.rows);
        float w = static_cast<float>(img.cols);
        float ratio = w / h;
        
        int resized_w;
        if (std::ceil(imgH * ratio) > target_w) {
            resized_w = target_w;
        } else {
            resized_w = static_cast<int>(std::ceil(imgH * ratio));
        }
        resized_w = std::max(resized_w, 1);
        
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(resized_w, imgH));
        
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3);
        float_img /= 255.0f;
        float_img -= 0.5f;
        float_img /= 0.5f;
        
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        
        int offset = b * imgC * imgH * target_w;
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < imgH; ++i) {
                const float* row = channels[c].ptr<float>(i);
                for (int j = 0; j < resized_w; ++j) {
                    batch_data[offset + c * imgH * target_w + i * target_w + j] = row[j];
                }
            }
        }
    }
    
    return {batch_data, {batch_size, imgC, imgH, target_w}};
}

// 批量分类预处理
inline std::pair<std::vector<float>, std::vector<int64_t>>
cls_preprocess_batch(const std::vector<cv::Mat>& imgs, int imgC, int imgH, int imgW) {
    int batch_size = static_cast<int>(imgs.size());
    std::vector<float> batch_data(batch_size * imgC * imgH * imgW, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        const cv::Mat& img = imgs[b];
        float h = static_cast<float>(img.rows);
        float w = static_cast<float>(img.cols);
        float ratio = w / h;
        int resized_w = std::min(imgW, static_cast<int>(std::ceil(imgH * ratio)));
        resized_w = std::max(resized_w, 1);
        
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(resized_w, imgH));
        
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3);
        float_img /= 255.0f;
        float_img -= 0.5f;
        float_img /= 0.5f;
        
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);
        
        int offset = b * imgC * imgH * imgW;
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < imgH; ++i) {
                const float* row = channels[c].ptr<float>(i);
                for (int j = 0; j < resized_w; ++j) {
                    batch_data[offset + c * imgH * imgW + i * imgW + j] = row[j];
                }
            }
        }
    }
    
    return {batch_data, {batch_size, imgC, imgH, imgW}};
}

} // namespace preprocess
} // namespace onnxocr
