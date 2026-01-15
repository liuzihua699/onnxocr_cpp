#pragma once

#include <vector>
#include <string>
#include <array>
#include <opencv2/opencv.hpp>

namespace onnxocr {

// 四边形框 [4个点, 每个点2个坐标]
using Box = std::array<cv::Point2f, 4>;

// OCR识别结果
struct RecResult {
    std::string text;
    float score;
};

// 单个文本行的完整结果
struct TextLine {
    Box box;
    std::string text;
    float score;
};

// 完整OCR结果
using OCRResult = std::vector<TextLine>;

// 分类结果
struct ClsResult {
    std::string label;
    float score;
};

} // namespace onnxocr
