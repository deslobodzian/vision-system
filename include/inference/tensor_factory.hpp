#ifndef VISION_SYSTEM_TENSOR_FACTORY_HPP
#define VISION_SYSTEM_TENSOR_FACTORY_HPP

#include <opencv2/opencv.hpp>
#include "tensor.hpp"
#include <onnxruntime_cxx_api.h>


template <typename T>
class TensorFactory {
public:
    /* Create a tensor from OpenCV Mat */
    static Tensor<T> from_cv_mat(const cv::Mat& mat, Device device = Device::CPU) {
        if (device != Device::CPU) {
            throw std::runtime_error("Tensor creation from cv::Mat only supports CPU for now.");
        }

        if (!mat.isContinuous()) {
            throw std::runtime_error("cv::Mat must be continuous.");
        }

        Shape shape = {mat.rows, mat.cols, mat.channels()};
        return Tensor<T>(reinterpret_cast<T*>(mat.data), shape, Device::CPU);
    }

    /* Create a tensor from Ort::Value */
    static Tensor<T> from_ort_value(Ort::Value& ort_value, Device device = Device::CPU) {
        if (!ort_value.IsTensor()) {
            throw std::runtime_error("The provided Ort::Value is not a tensor.");
        }

        auto tensor_info = ort_value.GetTensorTypeAndShapeInfo();
        auto element_type = tensor_info.GetElementType();

        if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && typeid(T) == typeid(float)) {
            throw std::runtime_error("Type mismatch between Ort::Value and Tensor.");
        }

        auto shape = tensor_info.GetShape();
        Shape tensor_shape(shape.begin(), shape.end());

        T* data_ptr = ort_value.GetTensorMutableData<T>();
        return Tensor<T>(data_ptr, tensor_shape, device);
    }

    static Ort::Value to_ort_value(const Tensor<T>& tensor, Ort::MemoryInfo& memory_info) {
        if (tensor.device() != Device::CPU) {
            throw std::runtime_error("ONNX Runtime tensor currently only supported on CPU");
        }

        auto shape = tensor.shape();
        std::vector<int64_t> ort_shape(shape.begin(), shape.end());

        return Ort::Value::CreateTensor<T>(
            memory_info,
            tensor.data(),
            tensor.size(),
            ort_shape.data(),
            ort_shape.size()
        );
    }    
};

#endif /* VISION_SYSTEM_TENSOR_FACTORY_HPP */

