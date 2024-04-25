//
// Created by deslobodzian on 11/23/22.
//
#include "april_tag_runner.hpp"
#include "april_tag_array_generated.h"
#include "utils/april_tag_utils.hpp"
#include "utils/logger.hpp"
#include "utils/timer.h"

AprilTagRunner::AprilTagRunner(std::shared_ptr<TaskManager> manager,
                               double period, const std::string &name,
                               const std::shared_ptr<ZmqManager> zmq_manager)
    : Task(manager, period, name), zmq_manager_(zmq_manager),
      use_detection_(false) {
#ifdef WITH_CUDA
  // Dennis's camera: 47502321
  // Outliers's camera: 41535987
  cfg_.serial_number = 41535987;
  // cfg_.serial_number = 47502321;
  cfg_.res = sl::RESOLUTION::SVGA;
  // cfg_.res = sl::RESOLUTION::VGA;
  cfg_.sdk_verbose = false;
  cfg_.enable_tracking = false;
  cfg_.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
  cfg_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  cfg_.max_depth = 20;
  cfg_.min_depth = 0.3f;
  cfg_.async_grab = true;
  cfg_.default_memory = sl::MEM::GPU;

  sl::Resolution res = sl::getResolution(cfg_.res);
  res.height = res.height / 2;
  res.width = res.width / 2;

  cfg_.depth_resolution = res;
  camera_.configure(cfg_);
  camera_.open();

  sl::Resolution display_resolution = camera_.get_resolution();

  uint32_t img_height = display_resolution.height;
  uint32_t img_width = display_resolution.width;
  constexpr uint32_t tile_size = 4;
  constexpr cuAprilTagsFamily tag_family = NVAT_TAG36H11;
  constexpr float tag_dim = 0.16f;
  constexpr int decimate = 2;
  tag_detector_.init_detector(img_width, img_height, tile_size, tag_family,
                              tag_dim, decimate);
#endif
}

void AprilTagRunner::init() {
  LOG_INFO("Initializing [AprilTagRunner]");
#ifdef WITH_CUDA
  camera_.enable_tracking();
#endif /* WITH_CUDA */
}

void AprilTagRunner::run() {
  Timer t;
  t.start();

  // Use this to bypass message
  use_detection_ = false;

#ifdef WITH_CUDA
  if (!use_detection_) {
    LOG_DEBUG("Using apriltag detection");
    camera_.fetch_measurements(MeasurementType::IMAGE_AND_POINT_CLOUD);
    auto zed_tags =
        AprilTagUtils::calculate_zed_apriltags(camera_, tag_detector_);
    std::vector<flatbuffers::Offset<Messages::AprilTag>> april_tag_offsets;

    auto &builder = zmq_manager_->get_publisher("FrontZed").get_builder();

    auto current_ms = t.get_ms();
    for (const auto &tag : zed_tags) {
      auto april_tag = Messages::CreateAprilTag(
          builder, tag.tag_id, tag.center.x, tag.center.y, tag.center.z,
          tag.orientation.ow, tag.orientation.ox, tag.orientation.oy,
          tag.orientation.oz,
          current_ms // now just how long processing takes for latency (will
                     // roughly be 20ms)
      );
      april_tag_offsets.push_back(april_tag);
    }

    auto tags_vector = builder.CreateVector(april_tag_offsets);
    auto april_tags_array = Messages::CreateAprilTagArray(builder, tags_vector);

    builder.Finish(april_tags_array);
    zmq_manager_->get_publisher("FrontZed")
        .publish_prebuilt("AprilTags", builder.GetBufferPointer(),
                          builder.GetSize());
    current_ms = t.get_ms();
    LOG_DEBUG("Zed pipline took: ", current_ms, " ms");
  } else {
    LOG_DEBUG("Using notes detection only not running tags to save resources");
  }
#endif
}

AprilTagRunner::~AprilTagRunner() {
#ifdef WITH_CUDA
  camera_.close();
#endif /* WITH_CUDA */
}
