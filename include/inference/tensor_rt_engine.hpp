#ifdef __CUDACC__
#ifndef VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP
#define VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP

#include <string>
#include "i_inference_engine.hpp"
#include "tensor.hpp"

using namespace nvinfer1;

struct OptimDim {
    nvinfer1::Dims4 size;
    std::string tensor_name;

    bool setFromString(std::string &arg) {
        // "images:1x3x512x512"
        std::vector<std::string> v_ = split_str(arg, ":");
        if (v_.size() != 2) return true;

        std::string dims_str = v_.back();
        std::vector<std::string> v = split_str(dims_str, "x");

        size.nbDims = 4;
        // assuming batch is 1 and channel is 3
        size.d[0] = 1;
        size.d[1] = 3;

        if (v.size() == 2) {
            size.d[2] = stoi(v[0]);
            size.d[3] = stoi(v[1]);
        } else if (v.size() == 3) {
            size.d[2] = stoi(v[1]);
            size.d[3] = stoi(v[2]);
        } else if (v.size() == 4) {
            size.d[2] = stoi(v[2]);
            size.d[3] = stoi(v[3]);
        } else return true;

        if (size.d[2] != size.d[3]) std::cerr << "Warning only squared input are currently supported" << std::endl;

        tensor_name = v_.front();
        return false;
    }
};

inline bool readFile(std::string filename, std::vector<uint8_t> &file_content) {
    // open the file:
    std::ifstream instream(filename, std::ios::in | std::ios::binary);
    if (!instream.is_open()) return true;
    file_content = std::vector<uint8_t>((std::istreambuf_iterator<char>(instream)), std::istreambuf_iterator<char>());
    return false;
}

class TensorRTEngine: public IInferenceEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();
    static int build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile);
    void load_model(const std::string& model_path) override;
    Tensor<float> run_inference(const Tensor<float>& input_tensor) override;
    const Shape get_input_shape() const override;
    const Shape get_output_shape() const override;

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    Tensor<float> input_;
    Tensor<float> output_;
};

#endif /* VISION_SYSTEM_ONNXRUNTIME_INFERENCE_ENGINE_HPP */
#endif
