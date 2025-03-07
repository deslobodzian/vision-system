#ifdef CUDA
#pragma once

#include <flatbuffers/flatbuffers.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "logger.h"
#include "zmq_publisher.h"
#include "zed.h"

class ZedPublisher {
public:
    explicit ZedPublisher(const std::string& endpoint); 
    std::vector<uint8_t> compress_frame(const cv::Mat& frame, int quality = 50);
    void write_measurements(const ZedMeasurements& measurements);

private:
    ZmqPublisher pub_;
};
#endif /* CUDA */