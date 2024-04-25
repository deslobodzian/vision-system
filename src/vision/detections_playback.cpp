#ifdef WITH_CUDA
#include "vision/detections_playback.hpp"
#include "utils/april_tag_utils.hpp"
#include "utils/logger.hpp"
#include "vision/detection_utils.hpp"
#include <chrono>

DetectionsPlayback::DetectionsPlayback(const std::string &svo_file)
    : //    yolo_("yolov8s.engine", det_cfg) {
      zed_(svo_file) {

  det_cfg.nms_thres = 0.3;
  det_cfg.obj_thres = 0.3;
  detector_.configure(det_cfg);

  cfg.id_retention_time = 0.0f;
  cfg.prediction_timeout_s = 0.01f;
  cfg.enable_segmentation = false;
  cfg.detection_confidence_threshold = 0.2f;
  cfg.enable_tracking = false;

  zed_.configure(cfg);

  zed_.open();
  zed_.enable_tracking();
  zed_.enable_object_detection();

  display_resolution = zed_.get_svo_resolution();

  uint32_t img_height = display_resolution.height;
  uint32_t img_width = display_resolution.width;
  LOG_INFO("Playback detection res: {", img_width, ", ", img_height, "}");
  uint32_t tile_size = 4;
  cuAprilTagsFamily tag_family = NVAT_TAG36H11;
  float tag_dim = 0.16f;
  int decimate = 3;

  tag_detector_.init_detector(img_width, img_height, tile_size, tag_family,
                              tag_dim, decimate);

  video_writer.open(
      "output_video.avi", cv::CAP_FFMPEG,
      cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30,
      cv::Size(display_resolution.width, display_resolution.height), true);
}

DetectionsPlayback::~DetectionsPlayback() { zed_.close(); }

void DetectionsPlayback::detect() {
  bool running_ = true;
  zed_.set_memory_type(sl::MEM::GPU);
  auto start = std::chrono::high_resolution_clock::now();
  while (running_) {
    zed_.fetch_measurements(MeasurementType::IMAGE_AND_POINT_CLOUD);
    auto ret = zed_.get_grab_state();
    LOG_DEBUG("Zed image data type: ", zed_.get_left_image().getDataType());
    detector_.detect_objects(zed_);
    LOG_DEBUG("Detecting Tags");
    auto detectedTags =
        tag_detector_.detect_april_tags_in_sl_image(zed_.get_left_image());
    // this will calculate tags twice
    auto zed_detected_tags =
        AprilTagUtils::calculate_zed_apriltags(zed_, tag_detector_);

    //        auto zed_detected_tags =
    //        tag_detector_.calculate_zed_apriltag(zed_.get_point_cloud(),
    //        zed_.get_normals(), detectedTags);
    zed_.fetch_measurements(MeasurementType::OBJECTS);
    // auto detections = yolo_.predict(zed_.get_left_image());
    auto err = left_sl.setFrom(zed_.get_left_image(), sl::COPY_TYPE::GPU_CPU);

    if (err == sl::ERROR_CODE::SUCCESS) {
      left_cv = slMat_to_cvMat(left_sl);
    } else {
      LOG_ERROR("Failed to update CPU from GPU: ", err);
    }
    cv::imwrite("left_cv.png", left_cv);

    const sl::Objects &objects = zed_.retrieve_objects();
    LOG_DEBUG("Zed objects detected: ", objects.object_list.size());

    for (size_t j = 0; j < objects.object_list.size(); j++) {
      cv::Rect r = sl_cvt_rect(objects.object_list[j].bounding_box_2d);
      cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(left_cv,
                  std::to_string((int)objects.object_list[j].raw_label),
                  cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                  cv::Scalar(0xFF, 0xFF, 0xFF), 2);
      sl::float3 position =
          objects.object_list[j].position; // Access the XYZ coordinates
      std::string xyz_text = "XYZ: " + std::to_string(position.x) + ", " +
                             std::to_string(position.y) + ", " +
                             std::to_string(position.z);
      cv::putText(
          left_cv, xyz_text,
          cv::Point(r.x, r.y + r.height +
                             15), // Positioning the text below the bounding box
          cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0x00, 0xFF, 0xFF), 2);
    }
    for (const auto &tag : detectedTags) {
      LOG_INFO("Tag ID: ", tag.id);
      for (int i = 0; i < 4; ++i) {
        LOG_INFO("Corner ", i, ",: {", tag.corners[i].x, ", ", tag.corners[i].y,
                 "}");
        cv::line(
            left_cv, cv::Point(tag.corners[i].x, tag.corners[i].y),
            cv::Point(tag.corners[(i + 1) % 4].x, tag.corners[(i + 1) % 4].y),
            cv::Scalar(0, 255, 0), 2);
      }
      cv::putText(left_cv, std::to_string(tag.id),
                  cv::Point(tag.corners[0].x, tag.corners[0].y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    if (left_cv.type() == CV_8UC3) {
    } else if (left_cv.type() == CV_8UC4) {
      cv::Mat threeChannelMat;
      cv::cvtColor(left_cv, threeChannelMat, cv::COLOR_BGRA2BGR);
      left_cv = threeChannelMat;
    } else {
      cv::cvtColor(left_cv, left_cv, cv::COLOR_GRAY2BGR);
    }
    if (!left_cv.empty()) {
      if (video_writer.isOpened()) {
        video_writer.write(left_cv);
      } else {
        LOG_ERROR("Video writer not open!");
      }
    } else {
      LOG_ERROR("Frame empty");
    }
    if (ret == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
      export_video();
      running_ = false;
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(stop - start).count();
  LOG_INFO("Detections playback took: ", elapsed, " seconds");
}
void DetectionsPlayback::export_video() {
  if (video_writer.isOpened()) {
    LOG_INFO("Released writer");
    video_writer.release();
  }
}
#endif /* WITH_CUDA */
