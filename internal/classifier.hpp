#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <memory>
#include "config.hpp"
#include "types.hpp"
#include "onnx_session.hpp"
#include "preprocess.hpp"
#include "cls_postprocess.hpp"

namespace onnxocr {

class TextClassifier {
public:
    TextClassifier(const Config& config) : config_(config) {
        session_.init(config.cls_model_path, config.use_gpu, config.gpu_id);
        postprocess_ = std::make_unique<postprocess::ClsPostProcess>(config.label_list);
    }
    
    std::pair<std::vector<cv::Mat>, std::vector<ClsResult>> 
    operator()(std::vector<cv::Mat> img_list) {
        if (img_list.empty()) {
            return {img_list, {}};
        }
        
        int img_num = static_cast<int>(img_list.size());
        std::vector<ClsResult> cls_res(img_num, {"", 0.0f});
        
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
        
        int batch_num = config_.cls_batch_num;
        int imgC = config_.cls_image_shape[0];
        int imgH = config_.cls_image_shape[1];
        int imgW = config_.cls_image_shape[2];
        
        for (int beg = 0; beg < img_num; beg += batch_num) {
            int end = std::min(img_num, beg + batch_num);
            int actual_batch = end - beg;
            
            // 收集batch内的图像
            std::vector<cv::Mat> batch_imgs;
            batch_imgs.reserve(actual_batch);
            for (int i = beg; i < end; ++i) {
                batch_imgs.push_back(img_list[indices[i]]);
            }
            
            // 批量预处理
            auto [batch_data, batch_shape] = preprocess::cls_preprocess_batch(
                batch_imgs, imgC, imgH, imgW);
            
            // 推理
            auto outputs = session_.run(batch_data, batch_shape);
            
            // 获取输出
            const float* output_data = outputs[0].GetTensorData<float>();
            auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // 后处理
            auto results = (*postprocess_)(output_data, 
                                            static_cast<int>(output_shape[0]),
                                            static_cast<int>(output_shape[1]));
            
            // 还原顺序并旋转
            for (int i = 0; i < static_cast<int>(results.size()); ++i) {
                int idx = indices[beg + i];
                cls_res[idx] = results[i];
                
                // 如果是180度且置信度高，旋转图片
                if (results[i].label.find("180") != std::string::npos &&
                    results[i].score > config_.cls_thresh) {
                    cv::rotate(img_list[idx], img_list[idx], cv::ROTATE_180);
                }
            }
        }
        
        return {img_list, cls_res};
    }

private:
    Config config_;
    OnnxSession session_;
    std::unique_ptr<postprocess::ClsPostProcess> postprocess_;
};

} // namespace onnxocr
