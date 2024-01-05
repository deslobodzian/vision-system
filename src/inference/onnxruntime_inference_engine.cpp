#include "inference/onnxruntime_inference_engine.hpp"
#include "inference/tensor_factory.hpp"
#include "utils/logger.hpp"
#include <opencv2/imgcodecs.hpp>

std::string tensor_shape_to_string(const std::vector<int64_t>& input_tensor) {
    std::string tensor_shape = "Tensor shape is [";
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        tensor_shape += std::to_string(input_tensor[i]);
        if (i < input_tensor.size() - 1) {
            tensor_shape += ", ";
        }
    }
    tensor_shape += "]";
    return tensor_shape;
}

void print_ort_tensor(const Ort::Value& tensor) {
    // Check if the tensor contains data
    if (!tensor.IsTensor()) {
        std::cerr << "The provided Ort::Value is not a tensor." << std::endl;
        return;
    }

    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    auto element_type = tensor_info.GetElementType();
    auto shape = tensor_info.GetShape();
    size_t total_num_elements = tensor_info.GetElementCount();

    std::cout << "Tensor Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* float_data = tensor.GetTensorData<float>();
        for (size_t i = 0; i < total_num_elements; ++i) {
            std::cout << float_data[i] << " ";
            if ((i + 1) % shape.back() == 0) std::cout << std::endl; // New line for each row in a 2D tensor
        }
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* int_data = tensor.GetTensorData<int64_t>();
        for (size_t i = 0; i < total_num_elements; ++i) {
            std::cout << int_data[i] << " ";
            if ((i + 1) % shape.back() == 0) std::cout << std::endl;
        }
    } else {
        std::cerr << "Unsupported tensor data type." << std::endl;
    }

    std::cout << std::endl;
}

ONNXRuntimeInferenceEngine::ONNXRuntimeInferenceEngine() : env_(ORT_LOGGING_LEVEL_WARNING, "Inference") {}

void ONNXRuntimeInferenceEngine::load_model(const std::string& model_path) {
    cv::Mat mat = cv::imread("bus.jpg");
    cv::Mat x = cv::imread("bus.jpg");
    Tensor<float> test1 = TensorFactory<float>::from_cv_mat(x);
    Tensor<float> test = TensorFactory<float>::from_cv_mat(mat);
    LOG_INFO(test.print_shape());

    memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );

    std::vector<std::string> providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        LOG_DEBUG("Provider available: ", provider);
    }

    Ort::Value onnx_tensor = TensorFactory<float>::to_ort_value(test, memory_info_);

    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1); // just for now 1 thread
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // might as well optimize :)

    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    size_t input_node_size = session_.GetInputCount();
    size_t output_node_size = session_.GetOutputCount();

    printf("Input node size: %zu\n", input_node_size);
    printf("Output node size: %zu\n", output_node_size);

    input_node_name_ = session_.GetInputNameAllocated(0, allocator_).get();
    LOG_INFO("Input name: ", input_node_name_);
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();
    input_shape_ = input_tensor_info.GetShape();
    LOG_INFO(tensor_shape_to_string(input_dims));
    output_node_name_ = session_.GetOutputNameAllocated(0, allocator_).get();
    LOG_INFO("Output name: ", output_node_name_);

    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = output_tensor_info.GetShape();
    LOG_INFO(tensor_shape_to_string(output_shape_));
}

Tensor<float> ONNXRuntimeInferenceEngine::run_inference(const Tensor<float>& input_tensor) {
    LOG_DEBUG("Running Inference");
    // For now batchsize is always 1, so this doesn't need to be an array, but in the future this might change.
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(TensorFactory<float>::to_ort_value(input_tensor, memory_info_));

    const char* input_node_name_cstr = input_node_name_.c_str();
    const char* output_node_name_cstr = output_node_name_.c_str();

    std::vector<Ort::Value> output_tensor = session_.Run(
        Ort::RunOptions{nullptr},
        &input_node_name_cstr,
        input_tensors.data(),
        1, // batch size of 1
        &output_node_name_cstr,
        1
    );
    Tensor output = TensorFactory<float>::from_ort_value(output_tensor.at(0));
    return output;
}

const Shape ONNXRuntimeInferenceEngine::get_input_shape() const {
    return input_shape_;
}

const Shape ONNXRuntimeInferenceEngine::get_output_shape() const {
    return output_shape_;
}
