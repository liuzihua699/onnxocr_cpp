#pragma once

#include <vector>
#include <string>
#include "types.hpp"

namespace onnxocr {
namespace postprocess {

class ClsPostProcess {
public:
    ClsPostProcess(const std::vector<std::string>& label_list)
        : label_list_(label_list) {}
    
    std::vector<ClsResult> operator()(const float* preds, int batch_size, int num_classes) const {
        std::vector<ClsResult> results;
        results.reserve(batch_size);
        
        for (int b = 0; b < batch_size; ++b) {
            const float* pred = preds + b * num_classes;
            
            // argmax
            int max_idx = 0;
            float max_val = pred[0];
            for (int c = 1; c < num_classes; ++c) {
                if (pred[c] > max_val) {
                    max_val = pred[c];
                    max_idx = c;
                }
            }
            
            std::string label;
            if (max_idx < static_cast<int>(label_list_.size())) {
                label = label_list_[max_idx];
            } else {
                label = std::to_string(max_idx);
            }
            
            results.push_back({label, max_val});
        }
        
        return results;
    }

private:
    std::vector<std::string> label_list_;
};

} // namespace postprocess
} // namespace onnxocr
