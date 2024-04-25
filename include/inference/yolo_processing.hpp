#ifndef VISION_SYSTEM_YOLO_PREPROCESS
#define VISION_SYSTEM_YOLO_PREPROCESS

/* CPU Pre-processing for yolo */

#include "tensor.hpp"
#include <opencv2/opencv.hpp>
#include <type_traits>

template <typename MatType>
inline void cpu_preprocess(const MatType& mat, Tensor<float>& input_tensor) {
  Shape s = input_tensor.shape();
  int input_w = s.at(2);
  int input_h = s.at(3);

  int image_w;
  int image_h;
  if constexpr (std::is_same_v<MatType, cv::Mat>) {
    LOG_INFO("Using cv Mat");
    image_w = mat.cols;
    image_h = mat.rows;
  } else {
    LOG_INFO("Not cv Mat");
  }

  float scale = std::min(static_cast<float>(input_w) / image_w,
                         static_cast<float>(input_h) / image_h);

  int scaled_w = static_cast<int>(image_w * scale);
  int scaled_h = static_cast<int>(image_w * scale);

  cv::Mat resized;
  cv::resize(mat, resized, cv::Size(scaled_w, scaled_h), 0, 0,
             cv::INTER_LINEAR);
  cv::Mat output(input_w, input_h, CV_8UC3, cv::Scalar(128, 128, 128));

  cv::Rect roi((input_w - scaled_w) / 2, (input_h - scaled_h) / 2, scaled_w,
               scaled_h);
  resized.copyTo(output(roi));

  input_tensor.to_cpu();
  input_tensor.copy(output.data,
                    Shape{output.cols, output.rows, output.channels()});

  // CWH -> WHC
  input_tensor.scale(1.f / 255.f);  // remember to normalize!!! I should add
                                    // this to Tensor such that its .normalize()
  input_tensor.reshape({1, 3, input_h, input_w});
}

#endif /* VISION_SYSTEM_YOLO_PREPROCESS */
