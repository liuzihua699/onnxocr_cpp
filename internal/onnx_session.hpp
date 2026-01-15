#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace onnxocr {

class OnnxSession {
public:
    OnnxSession() : env_(ORT_LOGGING_LEVEL_WARNING, "onnxocr") {}
    
    ~OnnxSession() = default;
    
    void init(const std::string& model_path, bool use_gpu = false, int gpu_id = 0) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
#ifdef USE_CUDA
        if (use_gpu) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = gpu_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
#else
        (void)use_gpu;
        (void)gpu_id;
#endif
        
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
        }
        
        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(name_ptr.get()));
        }
        for (const auto& name : input_names_) {
            input_names_ptr_.push_back(name.c_str());
        }
        
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(name_ptr.get()));
        }
        for (const auto& name : output_names_) {
            output_names_ptr_.push_back(name.c_str());
        }
    }
    
    std::vector<Ort::Value> run(const std::vector<float>& input_data, 
                                 const std::vector<int64_t>& input_shape) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        return session_->Run(
            Ort::RunOptions{nullptr},
            input_names_ptr_.data(),
            &input_tensor, 1,
            output_names_ptr_.data(),
            output_names_ptr_.size()
        );
    }
    
    // 批量推理（多个输入tensor）
    std::vector<Ort::Value> run_batch(const std::vector<std::vector<float>>& input_datas,
                                       const std::vector<std::vector<int64_t>>& input_shapes) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < input_datas.size(); ++i) {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(input_datas[i].data()),
                input_datas[i].size(),
                input_shapes[i].data(),
                input_shapes[i].size()
            ));
        }
        
        return session_->Run(
            Ort::RunOptions{nullptr},
            input_names_ptr_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_ptr_.data(),
            output_names_ptr_.size()
        );
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;
};

} // namespace onnxocr
