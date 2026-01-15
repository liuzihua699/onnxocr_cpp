#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "config.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "detector.hpp"
#include "recognizer.hpp"
#include "classifier.hpp"

namespace onnxocr {

class TextSystem {
public:
    TextSystem(const Config& config) : config_(config) {
        detector_ = std::make_unique<TextDetector>(config);
        recognizer_ = std::make_unique<TextRecognizer>(config);
        
        if (config.use_angle_cls) {
            classifier_ = std::make_unique<TextClassifier>(config);
        }
    }
    
    // 完整OCR：检测 + 分类(可选) + 识别
    OCRResult operator()(const cv::Mat& img, bool use_cls = true) {
        if (img.empty()) {
            return {};
        }
        
        // 检测
        auto boxes = (*detector_)(img);
        
        if (boxes.empty()) {
            return {};
        }
        
        // 排序
        boxes = utils::sorted_boxes(boxes);
        
        // 裁剪文本区域
        std::vector<cv::Mat> img_crop_list;
        img_crop_list.reserve(boxes.size());
        for (const auto& box : boxes) {
            cv::Mat crop = utils::get_rotate_crop_image(img, box);
            if (!crop.empty()) {
                img_crop_list.push_back(crop);
            }
        }
        
        if (img_crop_list.empty()) {
            return {};
        }
        
        // 分类（可选）
        if (config_.use_angle_cls && use_cls && classifier_) {
            auto [rotated_imgs, cls_results] = (*classifier_)(img_crop_list);
            img_crop_list = std::move(rotated_imgs);
        }
        
        // 识别
        auto rec_results = (*recognizer_)(img_crop_list);
        
        // 组装结果
        OCRResult results;
        results.reserve(boxes.size());
        
        size_t min_size = std::min(boxes.size(), rec_results.size());
        for (size_t i = 0; i < min_size; ++i) {
            if (rec_results[i].score >= config_.drop_score) {
                results.push_back({boxes[i], rec_results[i].text, rec_results[i].score});
            }
        }
        
        return results;
    }
    
    // 仅检测
    std::vector<Box> detect(const cv::Mat& img) {
        if (img.empty()) {
            return {};
        }
        return (*detector_)(img);
    }
    
    // 仅识别（输入已裁剪的文本图像列表）
    std::vector<RecResult> recognize(const std::vector<cv::Mat>& img_list) {
        return (*recognizer_)(img_list);
    }
    
    // 仅分类
    std::pair<std::vector<cv::Mat>, std::vector<ClsResult>> classify(std::vector<cv::Mat> img_list) {
        if (classifier_) {
            return (*classifier_)(std::move(img_list));
        }
        return {img_list, {}};
    }

private:
    Config config_;
    std::unique_ptr<TextDetector> detector_;
    std::unique_ptr<TextRecognizer> recognizer_;
    std::unique_ptr<TextClassifier> classifier_;
};

} // namespace onnxocr
