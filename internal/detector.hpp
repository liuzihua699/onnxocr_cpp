#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "config.hpp"
#include "types.hpp"
#include "onnx_session.hpp"
#include "preprocess.hpp"
#include "db_postprocess.hpp"
#include "utils.hpp"

namespace onnxocr {

class TextDetector {
public:
    TextDetector(const Config& config) : config_(config) {
        session_.init(config.det_model_path, config.use_gpu, config.gpu_id);
        
        postprocess_ = std::make_unique<postprocess::DBPostProcess>(
            config.det_db_thresh,
            config.det_db_box_thresh,
            1000,
            config.det_db_unclip_ratio,
            config.use_dilation,
            config.det_db_score_mode,
            config.det_box_type
        );
    }
    
    std::vector<Box> operator()(const cv::Mat& img) {
        if (img.empty()) {
            return {};
        }
        
        // 预处理
        auto prep_result = preprocess::det_preprocess(
            img, config_.det_limit_side_len, config_.det_limit_type);
        
        // 推理
        auto outputs = session_.run(prep_result.data, prep_result.shape);
        
        // 获取输出数据
        const float* output_data = outputs[0].GetTensorData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int height = static_cast<int>(output_shape[2]);
        int width = static_cast<int>(output_shape[3]);
        
        // 后处理
        auto boxes = (*postprocess_)(output_data, height, width,
                                      prep_result.src_h, prep_result.src_w);
        
        // 过滤和规整
        std::vector<Box> filtered_boxes;
        for (auto& box : boxes) {
            // 顺时针排序
            box = utils::order_points_clockwise(box);
            // 裁剪到图像边界
            box = utils::clip_box(box, img.rows, img.cols);
            
            // 检查尺寸
            float w = static_cast<float>(cv::norm(box[0] - box[1]));
            float h = static_cast<float>(cv::norm(box[0] - box[3]));
            if (w > 3 && h > 3) {
                filtered_boxes.push_back(box);
            }
        }
        
        return filtered_boxes;
    }

private:
    Config config_;
    OnnxSession session_;
    std::unique_ptr<postprocess::DBPostProcess> postprocess_;
};

} // namespace onnxocr
