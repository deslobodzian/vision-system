#ifdef CUDA
#include "zed_publisher.h"
#include "zed_utils.h"
#include "image_generated.h"
#include <chrono>

ZedPublisher::ZedPublisher(const std::string& endpoint) : pub_(endpoint) {
}

std::vector<uint8_t> ZedPublisher::compress_frame(const cv::Mat& frame, int quality) {
    std::vector<uint8_t> buf;
    std::vector<int> params = {};
    int ret = 0;
    if (frame.channels() == 1) { // Depth image
        params = {cv::IMWRITE_PNG_COMPRESSION, 0}; // Lossless
        ret = cv::imencode(".png", frame, buf, params);
    } else { // RGB image
        params = {cv::IMWRITE_JPEG_QUALITY, quality};
        ret = cv::imencode(".jpg", frame, buf, params);
    }
    return buf;
}

void ZedPublisher::write_measurements(const ZedMeasurements& measurements) {
    flatbuffers::FlatBufferBuilder builder(1024); // TODO: Private singleton
    // TODO: Check to make sure latest frame (maybe)
    sl::Pose sl_pose = measurements.camera_pose;
    sl::Translation t = sl_pose.getTranslation();
    sl::float3 angles = sl_pose.getEulerAngles(true);
    ZedImage::Pose pose = ZedImage::Pose(t.x, t.y, t.z, angles.x, angles.y, angles.z);

    sl::Mat left_image = measurements.left_image; 
    sl::Mat depth_image = measurements.depth_map;
    LOG_INFO("IMAGE: Height x Width {", left_image.getHeight(), " x ", left_image.getWidth(), "}");
    LOG_INFO("Depth IMAGE: Height x Width {", depth_image.getHeight(), " x ", depth_image.getWidth(), "}", " channels: ", depth_image.getChannels());

    // Maybe not needed as coversion might happen during sl::Mat -> cv::Mat?
    if (left_image.getMemoryType() == sl::MEM::GPU) { 
        left_image.updateCPUfromGPU();
    }

    cv::Mat cv_rgb = sl_to_cv(left_image);
    cv::Mat cv_depth = sl_to_cv(measurements.depth_map);
    std::vector<uint8_t> rgb_compressed = compress_frame(cv_rgb);

    const float* depth_ptr = reinterpret_cast<const float*>(cv_depth.data);
    std::vector<uint8_t> depth_raw(
        reinterpret_cast<const uint8_t*>(depth_ptr),
        reinterpret_cast<const uint8_t*>(depth_ptr + cv_depth.rows * cv_depth.cols)
    );

    builder.Clear();
    auto rgb_data = builder.CreateVector(rgb_compressed.data(), rgb_compressed.size());
    auto depth_data = builder.CreateVector(depth_raw.data(), depth_raw.size());


    auto frame = ZedImage::CreateFrame(
        builder,
        std::chrono::duration_cast<std::chrono::milliseconds> (
            std::chrono::system_clock::now().time_since_epoch()
        ).count(),
        cv_rgb.cols,
        cv_rgb.rows,
        rgb_data,
        depth_data,
        &pose
    );
    builder.Finish(frame);
    pub_.send_message("zed_image", builder.GetBufferPointer(), builder.GetSize());
}
#endif /* CUDA */
