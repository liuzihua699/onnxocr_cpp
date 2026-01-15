#pragma once

/**
 * OnnxOCR - Header-Only C++ OCR Library based on ONNX Runtime
 * 
 * 基于PaddleOCR的C++推理实现
 * 
 * 使用方法:
 * 
 *   #include "onnxocr/onnxocr.hpp"
 *   
 *   onnxocr::Config config;
 *   config.det_model_path = "models/det.onnx";
 *   config.rec_model_path = "models/rec.onnx";
 *   config.rec_char_dict_path = "models/ppocr_keys.txt";
 *   
 *   onnxocr::OCR ocr(config);
 *   
 *   cv::Mat img = cv::imread("test.jpg");
 *   auto results = ocr(img);
 *   
 *   for (const auto& line : results) {
 *       std::cout << line.text << " [" << line.score << "]" << std::endl;
 *   }
 */

// 核心类型和配置
#include "types.hpp"
#include "config.hpp"

// 工具函数
#include "utils.hpp"

// 预处理
#include "preprocess.hpp"

// 后处理
#include "db_postprocess.hpp"
#include "ctc_postprocess.hpp"
#include "cls_postprocess.hpp"

// ONNX会话
#include "onnx_session.hpp"

// 各模块
#include "detector.hpp"
#include "recognizer.hpp"
#include "classifier.hpp"

// 系统管理
#include "text_system.hpp"

namespace onnxocr {

// 便捷别名
using OCR = TextSystem;

} // namespace onnxocr
