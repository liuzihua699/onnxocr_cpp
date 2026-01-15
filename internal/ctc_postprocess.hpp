#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "types.hpp"

namespace onnxocr {
namespace postprocess {

class CTCLabelDecode {
public:
    CTCLabelDecode(const std::string& dict_path, bool use_space_char = true) {
        // 添加blank标记
        character_.push_back("blank");
        
        // 加载字典
        std::ifstream fin(dict_path, std::ios::binary);
        if (!fin.is_open()) {
            throw std::runtime_error("Failed to open dictionary file: " + dict_path);
        }
        
        std::string line;
        while (std::getline(fin, line)) {
            // 去除行尾的\r\n
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                line.pop_back();
            }
            if (!line.empty()) {
                character_.push_back(line);
            }
        }
        
        if (use_space_char) {
            character_.push_back(" ");
        }
    }
    
    std::vector<RecResult> operator()(const float* preds, int batch_size, 
                                      int seq_len, int num_classes) const {
        std::vector<RecResult> results;
        results.reserve(batch_size);
        
        for (int b = 0; b < batch_size; ++b) {
            const float* pred = preds + b * seq_len * num_classes;
            
            std::string text;
            std::vector<float> conf_list;
            int prev_idx = -1;
            
            // 逐时间步解码
            for (int t = 0; t < seq_len; ++t) {
                // argmax
                int max_idx = 0;
                float max_val = pred[t * num_classes];
                for (int c = 1; c < num_classes; ++c) {
                    if (pred[t * num_classes + c] > max_val) {
                        max_val = pred[t * num_classes + c];
                        max_idx = c;
                    }
                }
                
                // CTC解码：去重、去blank
                if (max_idx != 0 && max_idx != prev_idx) {
                    if (max_idx < static_cast<int>(character_.size())) {
                        text += character_[max_idx];
                        conf_list.push_back(max_val);
                    }
                }
                prev_idx = max_idx;
            }
            
            float score = 0.0f;
            if (!conf_list.empty()) {
                score = std::accumulate(conf_list.begin(), conf_list.end(), 0.0f) 
                        / static_cast<float>(conf_list.size());
            }
            
            results.push_back({text, score});
        }
        
        return results;
    }
    
    size_t get_char_count() const {
        return character_.size();
    }

private:
    std::vector<std::string> character_;
};

} // namespace postprocess
} // namespace onnxocr
