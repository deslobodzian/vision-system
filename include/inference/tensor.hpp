//
// Created by deslobodzian on 11/21/22.
//

#ifndef VISION_SYSTEM_TENSOR_HPP
#define VISION_SYSTEM_TENSOR_HPP

#include "utils/logger.hpp"
#include <initializer_list>
#include <memory>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

#ifdef WITH_CUDA
#include "cuda_utils.h"
#include <cuda_runtime.h>
#endif

template <typename T>
void cpu_deleter(T* p) {
#ifdef WITH_CUDA
  LOG_DEBUG("Pinned CPU memory free called");
  cudaFreeHost(p);
#else
  LOG_DEBUG("memory free called");
  delete[] p;
#endif
}

#ifdef WITH_CUDA
template <typename T>
void gpu_deleter(T* p) {
  LOG_DEBUG("GPU Free called");
  cudaFree(p);
}

#else
template <typename T>
void gpu_deleter(T* p) { /* CUDA not available */ }
#endif

enum class Device { CPU, GPU };
using Shape = std::vector<int64_t>;

template <typename T>
class Tensor {
 public:
  Tensor()
      : shape_(),
        device_(Device::CPU),
        is_gpu_data_valid_(false),
        cpu_data_(nullptr, cpu_deleter<T>),
        gpu_data_(nullptr, gpu_deleter<T>) {}

  Tensor(std::initializer_list<int64_t> shape, Device device);
  Tensor(Shape shape, Device device);
  Tensor(T* array, const Shape& shape, Device device);
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
  size_t size() const;

  T* gpu_data() const;
  T* data() const;

  // Incase gpu data is manipulated elsewhere
  void update_cpu_from_gpu();

  void reshape(const Shape& new_shape);
  void scale(T factor);

  void permute(const Shape& order);

  template <typename U>
  void copy(const Tensor<U>& other);

  template <typename U>
  void copy(const U* data, const Shape& source_shape);
  std::vector<size_t> calculate_strides(const Shape& shape);
  // TODO: Add the function in python that addes a dimension to the
  // tensor(forgot the name)
  std::string print_shape() const;
  std::string to_string() const;

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    return os << tensor.to_string();
  }

  Device device() const { return device_; }

 private:
  Shape shape_;
  Device device_;
  std::unique_ptr<T[], void (*)(T*)> cpu_data_;
  std::unique_ptr<T[], void (*)(T*)> gpu_data_;
  bool is_gpu_data_valid_;

  size_t calculate_size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1,
                           std::multiplies<size_t>());
  }
};

template <typename T>
Tensor<T>::Tensor(Shape shape, Device device)
    : shape_(std::move(shape)),
      device_(device),
      is_gpu_data_valid_(false),
      cpu_data_(nullptr, cpu_deleter<T>),
      gpu_data_(nullptr, gpu_deleter<T>) {
  allocate_memory();
}

template <typename T>
Tensor<T>::Tensor(std::initializer_list<int64_t> shape, Device device)
    : Tensor(Shape(shape), device) {}

template <typename T>
Tensor<T>::Tensor(T* array, const Shape& shape, Device device)
    : shape_(shape),
      device_(device),
      is_gpu_data_valid_(false),
      cpu_data_(nullptr, cpu_deleter<T>),
      gpu_data_(nullptr, gpu_deleter<T>) {
  allocate_memory();
  std::copy(array, array + calculate_size(), cpu_data_.get());
}

template <typename T>
template <typename U>
Tensor<T>::Tensor(const Tensor<U>& other)
    : shape_(other.shape()),
      device_(other.device()),
      is_gpu_data_valid_(false),
      cpu_data_(nullptr, cpu_deleter<T>),
      gpu_data_(nullptr, gpu_deleter<T>) {
  allocate_memory();
  std::copy(other.data(), other.data() + calculate_size(), cpu_data_.get());
}

template <typename T>
Tensor<T>::~Tensor() {
  free_memory();
}

template <typename T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept
    : shape_(std::move(other.shape_)),
      device_(other.device_),
      cpu_data_(std::move(other.cpu_data_)),
      gpu_data_(std::move(other.gpu_data_)),
      is_gpu_data_valid_(other.is_gpu_data_valid_) {}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    device_ = other.device_;
    cpu_data_ = std::move(other.cpu_data_);
    gpu_data_ = std::move(other.gpu_data_);
    is_gpu_data_valid_ = other.is_gpu_data_valid_;
  }
  return *this;
}

template <typename T>
void Tensor<T>::allocate_memory() {
  size_t total_size = calculate_size();
#ifdef WITH_CUDA
  T* new_cpu_data;
  // pinned memory for a bit more performance
  cudaMallocHost(&new_cpu_data, total_size * sizeof(T));
  cpu_data_.reset(new_cpu_data);
  if (device_ == Device::GPU) {
    T* gpu_data;
    cudaMalloc(&gpu_data, total_size * sizeof(T));
    gpu_data_.reset(gpu_data);
    is_gpu_data_valid_ = true;
  }
#else
  cpu_data_.reset(new T[total_size]);
#endif
}

template <typename T>
void Tensor<T>::free_memory() {
  cpu_data_.reset();
  gpu_data_.reset();
}

template <typename T>
void Tensor<T>::to_gpu() {
#ifdef WITH_CUDA
  if (device_ != Device::GPU) {
    if (!gpu_data_) {
      LOG_DEBUG("Allocating GPU data");
      T* gpu_data;
      CUDA_CHECK(cudaMalloc(&gpu_data, calculate_size() * sizeof(T)));
      gpu_data_.reset(gpu_data);
    }
    if (!is_gpu_data_valid_) {
      LOG_DEBUG("Data is not latest, copy data to GPU");
      CUDA_CHECK(cudaMemcpyAsync(gpu_data_.get(), cpu_data_.get(),
                                 calculate_size() * sizeof(T),
                                 cudaMemcpyHostToDevice));
      is_gpu_data_valid_ = true;
    }
    device_ = Device::GPU;
  }
#else
  LOG_ERROR("Cuda support not available");
#endif
}

template <typename T>
void Tensor<T>::to_cpu() {
#ifdef WITH_CUDA
  if (device_ != Device::CPU) {
    if (!gpu_data_) {
      throw std::runtime_error("GPU data not allocated.");
    }
    LOG_DEBUG("Transferring data to CPU");
    CUDA_CHECK(cudaMemcpyAsync(cpu_data_.get(), gpu_data_.get(),
                               calculate_size() * sizeof(T),
                               cudaMemcpyDeviceToHost));
    device_ = Device::CPU;
    is_gpu_data_valid_ = false;
  }
#endif
}

// force update cpu data from gpu
template <typename T>
void Tensor<T>::update_cpu_from_gpu() {
#ifdef WITH_CUDA
  LOG_DEBUG("Force update cpu from gpu data");
  CUDA_CHECK(cudaMemcpyAsync(cpu_data_.get(), gpu_data_.get(),
                             calculate_size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
  is_gpu_data_valid_ = false;
  device_ = Device::CPU;
#endif
}

template <typename T>
const Shape& Tensor<T>::shape() const {
  return shape_;
}

template <typename T>
size_t Tensor<T>::size() const {
  return calculate_size();
}
template <typename T>
T* Tensor<T>::gpu_data() const {
#ifdef WITH_CUDA
  return gpu_data_.get();
#else
  LOG_ERROR("No gpu, returning nullptr");
  return nullptr;
#endif
}
template <typename T>
T* Tensor<T>::data() const {
#ifdef WITH_CUDA
  if (device_ == Device::GPU) {
    LOG_DEBUG("Getting GPU data");
    return gpu_data_.get();
  }
#endif
  LOG_DEBUG("Getting CPU data");
  return cpu_data_.get();
}

template <typename T>
void Tensor<T>::reshape(const Shape& new_shape) {
  if (calculate_size() != std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                          std::multiplies<size_t>())) {
    throw std::invalid_argument(
        "New shape must have the same number of elements as the old shape.");
  }
  shape_ = new_shape;
}

template <typename T>
void Tensor<T>::scale(T factor) {
  LOG_DEBUG("Scaling tensor by: ", factor);
  size_t total_size = calculate_size();

  if (device_ == Device::CPU) {
    T* data_ptr = cpu_data_.get();
    for (size_t i = 0; i < total_size; ++i) {
      data_ptr[i] *= factor;
    }
  } else {
    LOG_ERROR("GPU support for scale not implemented, use CPU");
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
  LOG_DEBUG("Permuting tensor");
  if (order.size() != shape_.size()) {
    throw std::invalid_argument(
        "Permute order does not match tensor dimensions.");
  }

  for (auto idx : order) {
    if (idx < 0 || idx >= static_cast<int64_t>(shape_.size())) {
      throw std::invalid_argument("Invalid permute order.");
    }
  }

  Shape new_shape(shape_.size());
  for (size_t i = 0; i < order.size(); ++i) {
    new_shape[i] = shape_[order[i]];
  }
  std::vector<size_t> old_strides = calculate_strides(shape_);
  std::vector<size_t> new_strides = calculate_strides(new_shape);

  size_t total_size = calculate_size();
  std::unique_ptr<T[]> new_data(new T[total_size]);

  if (device_ == Device::CPU) {
    T* old_data = cpu_data_.get();
    for (size_t i = 0; i < total_size; ++i) {
      size_t old_idx = 0;
      size_t tmp = i;
      for (size_t j = 0; j < shape_.size(); ++j) {
        size_t axis = order[j];
        old_idx += (tmp / new_strides[j]) * old_strides[axis];
        tmp = tmp % new_strides[j];
      }
      new_data[i] = old_data[old_idx];
    }
    cpu_data_.reset(new_data.release());
  } else {
    LOG_ERROR("GPU support for permute not implemented, use CPU");
  }

  shape_ = new_shape;
}

template <typename T>
template <typename U>
void Tensor<T>::copy(const Tensor<U>& other) {
  LOG_DEBUG("Shape check in copy");
  if (this->shape() != other.shape()) {
    throw std::runtime_error("Tensor shapes do not match");
  }

  LOG_DEBUG("Calculating total elements in copy");
  size_t total_elements = this->calculate_size();

  LOG_DEBUG("Total elements: ", total_elements, " in copy");
  T* this_data = this->data();
  const U* other_data = other.data();

  if (!this_data || !other_data) {
    throw std::runtime_error("Null data pointer detected");
  }

  LOG_DEBUG("Copying data");
  for (size_t i = 0; i < total_elements; ++i) {
    this_data[i] = static_cast<T>(other_data[i]);
  }
  LOG_DEBUG("Copy completed");
}
template <typename T>
template <typename U>
void Tensor<T>::copy(const U* data, const Shape& source_shape) {
  if (device_ == Device::GPU) {
    LOG_ERROR("GPU support for copy not implemented, use CPU");
  }
  if (!data) {
    throw std::runtime_error("Null data pointer provided.");
  }

  size_t source_total_elements = std::accumulate(
      source_shape.begin(), source_shape.end(), 1, std::multiplies<size_t>());
  size_t dest_total_elements = calculate_size();
  if (source_total_elements != dest_total_elements) {
    throw std::runtime_error(
        "Total number of elements mismatch between source and destination.");
  }

  T* tensor_data = this->data();
  const size_t height = source_shape[0];
  const size_t width = source_shape[1];
  const size_t channels = source_shape[2];

  for (size_t c = 0; c < channels; ++c) {
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const size_t src_index = c + x * channels + y * width * channels;
        const size_t dest_index = c * width * height + y * width + x;
        tensor_data[dest_index] = static_cast<T>(data[src_index]);
      }
    }
  }
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
    LOG_ERROR("Data on GPU, transfer to CPU to access.");
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

  std::function<void(std::ostringstream&, const T*, size_t, size_t,
                     const std::vector<size_t>&)>
      print_array;
  print_array = [&](std::ostringstream& os, const T* data, size_t dim,
                    size_t total_dims,
                    const std::vector<size_t>& current_strides) {
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
        print_array(os, data + i * current_strides[dim], dim + 1, total_dims,
                    current_strides);
      }
      os << "]";
    }
  };

  print_array(oss, cpu_data_.get(), 0, shape_.size(), strides);

  return oss.str();
}

#endif  // VISION_SYSTEM_TENSOR_HPP
