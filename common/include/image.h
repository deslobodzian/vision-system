#pragma once

#include "device.h"

template <typename T> class Image {
public:
  // Create an image on specified device with OpenCV format height, width, channel
  Image(const Device &device, size_t height, size_t width, size_t channels)
      : height_(height), width_(width), channels_(channels),
        stride_(width_ * channels_ * sizeof(T)),
        buffer_(device, height_ * width_ * channels_) {
    LOG_DEBUG(
            "Creating image on ", device.name(),
            " H x W x C {", height, "x", width, "x", channels, "}"
    );
    buffer_.allocate();
  }

  size_t height() const { return height_; }
  size_t width() const { return width_; }
  size_t channels() const { return channels_; }
  /**
   * @return the stride of the Image in bytes, this is Row * Channels * sizeof(T)
  */
  size_t stride() const { return stride_; }
  size_t size() const { return buffer_.buffer_size(); }

private:
  size_t height_;
  size_t width_;
  size_t channels_;
  size_t stride_;
  Buffer<T> buffer_;
};
