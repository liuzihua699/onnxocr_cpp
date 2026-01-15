#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <memory>
#include "config.hpp"
#include "types.hpp"
#include "onnx_session.hpp"
#include "preprocess.hpp"
#include "ctc_postprocess.hpp"

namespace onnxocr {

class TextRecognizer {
public:
    TextRecognizer(const Config& config) : config_(config) {
        session_.init(config.rec_model_path, config.use_gpu, config.gpu_id);
        postprocess_ = std::make_unique<postprocess::CTCLabelDecode>(
            config.rec_char_dict_path, config.use_space_char);
    }
    
    std::vector<RecResult> operator()(const std::vector<cv::Mat>& img_list) {
        if (img_list.empty()) {
            return {};
        }
        
        int img_num = static_cast<int>(img_list.size());
        std::vector<RecResult> rec_res(img_num, {"", 0.0f});
        
        // 计算宽高比并排序
        std::vector<float> width_list;
        width_list.reserve(img_num);
        for (const auto& img : img_list) {
            if (img.rows > 0) {
                width_list.push_back(static_cast<float>(img.cols) / img.rows);
            } else {
                width_list.push_back(0.0f);
            }
        }
        
        std::vector<int> indices(img_num);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), 
                  [&](int a, int b) { return width_list[a] < width_list[b]; });
        
        int batch_num = config_.rec_batch_num;
        int imgC = config_.rec_image_shape[0];
        int imgH = config_.rec_image_shape[1];
        int imgW = config_.rec_image_shape[2];
        
        for (int beg = 0; beg < img_num; beg += batch_num) {
            int end = std::min(img_num, beg + batch_num);
            int actual_batch = end - beg;
            
            // 计算batch内最大宽高比
            float max_wh_ratio = static_cast<float>(imgW) / imgH;
            for (int i = beg; i < end; ++i) {
                max_wh_ratio = std::max(max_wh_ratio, width_list[indices[i]]);
            }
            
            int target_w = static_cast<int>(imgH * max_wh_ratio);
            target_w = std::max(target_w, 1);
            
            // 收集batch内的图像
            std::vector<cv::Mat> batch_imgs;
            batch_imgs.reserve(actual_batch);
            for (int i = beg; i < end; ++i) {
                batch_imgs.push_back(img_list[indices[i]]);
            }
            
            // 批量预处理
            auto [batch_data, batch_shape] = preprocess::rec_preprocess_batch(
                batch_imgs, imgC, imgH, target_w);
            
            // 推理
            auto outputs = session_.run(batch_data, batch_shape);
            
            // 获取输出
            const float* output_data = outputs[0].GetTensorData<float>();
            auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // 后处理
            auto results = (*postprocess_)(output_data, 
                                            static_cast<int>(output_shape[0]),
                                            static_cast<int>(output_shape[1]),
                                            static_cast<int>(output_shape[2]));
            
            // 还原顺序
            for (int i = 0; i < static_cast<int>(results.size()); ++i) {
                rec_res[indices[beg + i]] = results[i];
            }
        }
        
        return rec_res;
    }

private:
    Config config_;
    OnnxSession session_;
    std::unique_ptr<postprocess::CTCLabelDecode> postprocess_;
};

} // namespace onnxocr
