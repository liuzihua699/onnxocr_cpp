#pragma once

#include <string>
#include <vector>
#include <array>

namespace onnxocr {

struct Config {
    // 通用配置
    bool use_gpu = false;
    int gpu_id = 0;
    
    // 检测器配置
    std::string det_model_path = "models/det.onnx";
    int det_limit_side_len = 960;
    std::string det_limit_type = "max";  // "max" or "min"
    std::string det_box_type = "quad";   // "quad" or "poly"
    float det_db_thresh = 0.3f;
    float det_db_box_thresh = 0.6f;
    float det_db_unclip_ratio = 1.5f;
    bool use_dilation = false;
    std::string det_db_score_mode = "fast";
    
    // 识别器配置
    std::string rec_model_path = "models/rec.onnx";
    std::string rec_char_dict_path = "models/ppocr_keys_v1.txt";
    std::array<int, 3> rec_image_shape = {3, 48, 320};
    int rec_batch_num = 6;
    bool use_space_char = true;
    
    // 分类器配置
    bool use_angle_cls = false;
    std::string cls_model_path = "models/cls.onnx";
    std::array<int, 3> cls_image_shape = {3, 48, 192};
    int cls_batch_num = 6;
    float cls_thresh = 0.9f;
    std::vector<std::string> label_list = {"0", "180"};
    
    // 结果过滤
    float drop_score = 0.5f;
};

} // namespace onnxocr
