//
// Created by deslobodzian on 11/21/22.
//

#ifndef VISION_SYSTEM_TENSOR_HPP
#define VISION_SYSTEM_TENSOR_HPP

#include <vector>
#include <memory>
#include <numeric>
#include <initializer_list>
#include <stdexcept>
#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

template <typename T>
void cpu_deleter(T* p) { delete[] p; }

#ifdef __CUDACC__
template <typename T>
void gpu_deleter(T* p) { cudaFree(p); }
#else
template <typename T>
void gpu_deleter(T* p) { /* CUDA not available */ }
#endif

enum class Device { CPU, GPU };
using Shape = std::vector<int64_t>;

template <typename T>
class Tensor {
public:
    Tensor(std::initializer_list<int64_t> shape, Device device);
    Tensor(Shape shape, Device device);
    Tensor(T* array, const Shape& shape, Device device, bool owns_data = false);
    template <typename U>
    Tensor(const Tensor<U>& other);

    ~Tensor(); 

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void allocate_memory();
    void free_memory();

    void to_cpu();
    void to_gpu();

    const Shape& shape() const; 
    const size_t size() const;
    T* data() const;
    void reshape(const Shape& new_shape);
    void scale(T factor);

    void permute(const Shape& order);
    std::vector<size_t> calculate_strides(const Shape& shape);
    // TODO: Add the function in python that addes a dimension to the tensor(forgot the name)
    std::string print_shape() const;
    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        return os << tensor.to_string();
    }

    Device device() const {
        return device_;
    }

private:
    Shape shape_;
    Device device_;
    std::unique_ptr<T[], void (*)(T*)> data_;
    bool owns_data_;

    size_t calculate_size() const {
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
    }
};

template <typename T>
Tensor<T>::Tensor(std::initializer_list<int64_t> shape, Device device)
    : shape_(shape), device_(device), owns_data_(true),
      data_(nullptr, device == Device::CPU ? cpu_deleter<T> : gpu_deleter<T>) {
    allocate_memory();
}

template <typename T>
Tensor<T>::Tensor(Shape shape, Device device)
    : shape_(std::move(shape)), device_(device), owns_data_(true),
      data_(nullptr, device == Device::CPU ? cpu_deleter<T> : gpu_deleter<T>) {
    allocate_memory();
}

template <typename T>
Tensor<T>::Tensor(T* array, const Shape& shape, Device device, bool owns_data)
    : shape_(shape), device_(device), owns_data_(owns_data),
      data_(array, owns_data ? [](T* p) { delete[] p; } : [](T*) {}) {
    if (device != Device::CPU) {
        throw std::runtime_error("Non-CPU tensor construction from array not supported in this implementation.");
    }
}

template <typename T>
template <typename U>
Tensor<T>::Tensor(const Tensor<U>& other) 
    : shape_(other.shape()), device_(other.device()), 
      data_(nullptr, device_ == Device::CPU ? cpu_deleter<T> : gpu_deleter<T>), 
      owns_data_(true) {
    
    allocate_memory();
    
    const U* old_data = other.data();
    T* new_data = data_.get();
    size_t total_size = calculate_size();

    for (size_t i = 0; i < total_size; ++i) {
        new_data[i] = static_cast<T>(old_data[i]);
    }
}

template <typename T>
Tensor<T>::~Tensor() {
    free_memory();
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept
    : shape_(std::move(other.shape_)), device_(other.device_), data_(std::move(other.data_)) {
    other.data_ = nullptr;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        device_ = other.device_;
        data_ = std::move(other.data_);
        other.data_ = nullptr;
    }
    return *this;
}

template <typename T>
void Tensor<T>::allocate_memory() {
    size_t total_size = calculate_size();
    if (device_ == Device::CPU) {
        data_.reset(new T[total_size]);
    } else {
#ifdef __CUDACC__
        T* gpu_data;
        cudaMalloc(&gpu_data, total_size * sizeof(T));
        data_.reset(gpu_data);
#else
        throw std::runtime_error("CUDA support not available.");
#endif
    }
}

template <typename T>
void Tensor<T>::free_memory() {
    data_.reset();
}

template <typename T>
void Tensor<T>::to_cpu() {
    if (device_ == Device::GPU) {
#ifdef __CUDACC__
        T* cpu_data = new T[calculate_size()];
        cudaMemcpy(cpu_data, data_.get(), calculate_size() * sizeof(T), cudaMemcpyDeviceToHost);
        data_.reset(cpu_data);
        device_ = Device::CPU;
#else
        throw std::runtime_error("CUDA support not available.");
#endif
    }
}


template <typename T>
void Tensor<T>::to_gpu() {
    if (device_ == Device::CPU) {
#ifdef __CUDACC__
        T* gpu_data;
        cudaMalloc(&gpu_data, calculate_size() * sizeof(T));
        cudaMemcpy(gpu_data, data_.get(), calculate_size() * sizeof(T), cudaMemcpyHostToDevice);
        data_.reset(gpu_data);
        device_ = Device::GPU;
#else
        throw std::runtime_error("CUDA support not available.");
#endif
    }
}

template <typename T>
const Shape& Tensor<T>::shape() const {
    return shape_;
}

template <typename T>
const size_t Tensor<T>::size() const {
    return calculate_size();
}

template <typename T>
T* Tensor<T>::data() const {
    if (device_ == Device::GPU) {
        throw std::runtime_error("Data on GPU, transfer to CPU to access.");
    }
    return data_.get();
}

template <typename T>
void Tensor<T>::reshape(const Shape& new_shape) {
    if (calculate_size() != std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>())) {
        throw std::invalid_argument("New shape must have the same number of elements as the old shape.");
    }
    shape_ = new_shape;
}

template <typename T>
void Tensor<T>::scale(T factor) {
    if (device_ != Device::CPU) {
        throw std::runtime_error("Scaling on GPU not supported in this implementation.");
    }

    size_t total_size = calculate_size();
    T* data_ptr = data_.get();

    for (size_t i = 0; i < total_size; ++i) {
        data_ptr[i] *= factor;
    }
}

template <typename T>
std::vector<size_t> Tensor<T>::calculate_strides(const Shape& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template <typename T>
void Tensor<T>::permute(const Shape& order) {
    if (order.size() != shape_.size()) {
        throw std::invalid_argument("Permute order does not match tensor dimensions.");
    }

    for (auto idx : order) {
        if (idx < 0 || idx >= static_cast<int64_t>(shape_.size())) {
            throw std::invalid_argument("Invalid permute order.");
        }
    }

    Shape new_shape(shape_.size());
    std::vector<size_t> old_strides = calculate_strides(shape_);
    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = shape_[order[i]];
    }

    size_t total_size = calculate_size();
    T* new_data = new T[total_size];
    
    std::vector<size_t> new_strides = calculate_strides(new_shape);

    for (size_t i = 0; i < total_size; ++i) {
        size_t old_idx = 0;
        size_t tmp = i;
        for (size_t j = 0; j < shape_.size(); ++j) {
            size_t axis = order[j];
            old_idx += (tmp / new_strides[j]) * old_strides[axis];
            tmp = tmp % new_strides[j];
        }
        new_data[i] = data_[old_idx];
    }

    shape_ = new_shape;
    data_.reset(new_data);
}

template <typename T>
std::string Tensor<T>::print_shape() const {
    std::ostringstream oss;
    oss << "Tensor shape: (";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    oss << ")";
    return oss.str();
}

template <typename T>
std::string Tensor<T>::to_string() const {
    if (device_ == Device::GPU) {
        throw std::runtime_error("Data on GPU, transfer to CPU to access.");
    }

    std::ostringstream oss;
    oss << "Tensor shape: (";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
    }
    oss << ")\n";

    std::vector<size_t> strides(shape_.size(), 1);
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_[i + 1];
    }

    std::function<void(std::ostringstream&, const T*, size_t, size_t, const std::vector<size_t>&)> print_array;
    print_array = [&](std::ostringstream& os, const T* data, size_t dim, size_t total_dims, const std::vector<size_t>& current_strides) {
        if (dim == total_dims - 1) {
            os << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                os << data[i];
                if (i != shape_[dim] - 1) {
                    os << ", ";
                }
            }
            os << "]";
        } else {
            os << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i != 0) {
                    os << "\n" << std::string(dim * 2, ' ');
                }
                print_array(os, data + i * current_strides[dim], dim + 1, total_dims, current_strides);
            }
            os << "]";
        }
    };

    print_array(oss, data_.get(), 0, shape_.size(), strides);

    return oss.str();
}

#endif // VISION_SYSTEM_TENSOR_HPP