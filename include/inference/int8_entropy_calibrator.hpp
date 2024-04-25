#ifndef VISION_SYSTEM_INT8_ENTROPY_CALIBRATOR
#define VISION_SYSTEM_INT8_ENTROPY_CALIBRATOR
#include "NvInfer.h"
#include "preprocess_kernels.h"
#include "tensor.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

inline bool doesFileExist(const std::string& path) {
  return fs::exists(path);
}

inline std::vector<std::string> getFilesInDirectory(
    const std::string& directoryPath) {
  std::vector<std::string> files;
  for (const auto& entry : fs::directory_iterator(directoryPath)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path().string());
    }
  }
  return files;
}

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Int8EntropyCalibrator2(int batch_size,
                         const std::string& calibration_data_path,
                         const std::string& calibration_table_name,
                         const std::string& input_blob_name, const Shape& shape)
      : batch_size_(batch_size),
        calibration_data_path_(calibration_data_path),
        calibration_table_name_(calibration_table_name),
        input_blob_name_(input_blob_name),
        current_image_index_(0),
        data_tensor_(shape, Device::CPU) {
    // gpu allocation
    data_tensor_.to_gpu();

    data_tensor_.to_cpu();
    // Load calibration data file paths
    LOG_DEBUG("Loading calibration data");
    loadCalibrationData(calibration_data_path_);
    LOG_DEBUG("Loaded calibration data");
    cudaStreamCreate(&stream_);
  }

  int getBatchSize() const noexcept override { return batch_size_; }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override {
    if (current_image_index_ >= image_paths_.size()) {
      return false;  // No more data to process
    }

    // Assuming batch size is 1 for now
    LOG_DEBUG("Reading img: ", image_paths_[current_image_index_]);
    cv::Mat img =
        cv::imread(image_paths_[current_image_index_], cv::IMREAD_UNCHANGED);
    if (img.empty()) {
      std::cerr << "Error: Image could not be read." << std::endl;
      return false;
    }

    data_tensor_.to_gpu();
    preprocess_cv(img, data_tensor_, stream_);

    bindings[0] = data_tensor_.data();

    current_image_index_++;
    return true;
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    std::cout << "Searching for calibration cache: " << calibration_table_name_
              << std::endl;
    calibration_cache_.clear();
    std::ifstream input(calibration_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good()) {
      std::cout << "Reading calibration cache: " << calibration_table_name_
                << std::endl;
      std::copy(std::istream_iterator<char>(input),
                std::istream_iterator<char>(),
                std::back_inserter(calibration_cache_));
    }
    length = calibration_cache_.size();
    return length ? calibration_cache_.data() : nullptr;
  }

  void writeCalibrationCache(const void* cache,
                             size_t length) noexcept override {
    std::cout << "Writing calib cache: " << calibration_table_name_
              << " Size: " << length << " bytes" << std::endl;
    std::ofstream output(calibration_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }

 private:
  int batch_size_;
  std::string calibration_data_path_;
  std::string calibration_table_name_;
  std::string input_blob_name_;
  std::vector<std::string> image_paths_;
  std::vector<char> calibration_cache_;

  int current_image_index_;
  cudaStream_t stream_;
  Tensor<float> data_tensor_;

  bool read_cache_ = false;

  void loadCalibrationData(const std::string& calibDataDirPath) {
    // Check if the calibration data directory exists
    if (!doesFileExist(calibDataDirPath)) {
      throw std::runtime_error("Error, directory does not exist: " +
                               calibDataDirPath);
    }

    // Get all file paths in the calibration data directory
    image_paths_ = getFilesInDirectory(calibDataDirPath);

    // Check if there are enough images for the specified batch size
    if (image_paths_.size() < static_cast<size_t>(batch_size_)) {
      throw std::runtime_error(
          "Fewer calibration images than specified batch size!");
    }

    // Randomize the order of the calibration images
    std::shuffle(image_paths_.begin(), image_paths_.end(),
                 std::default_random_engine(std::random_device{}()));
    LOG_DEBUG("Loaded ", image_paths_.size(), " images");
  }
};
#endif /* VISION_SYSTEM_INT8_ENTROPY_CALIBRATOR */
