#ifdef WITH_CUDA

#ifndef VISION_SYSTEM_ZED_HPP
#define VISION_SYSTEM_ZED_HPP

#include "i_camera.hpp"
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <vector>

static int get_ocv_type(sl::MAT_TYPE type) {
  int cv_type = -1;
  switch (type) {
  case sl::MAT_TYPE::F32_C1:
    cv_type = CV_32FC1;
    break;
  case sl::MAT_TYPE::F32_C2:
    cv_type = CV_32FC2;
    break;
  case sl::MAT_TYPE::F32_C3:
    cv_type = CV_32FC3;
    break;
  case sl::MAT_TYPE::F32_C4:
    cv_type = CV_32FC4;
    break;
  case sl::MAT_TYPE::U8_C1:
    cv_type = CV_8UC1;
    break;
  case sl::MAT_TYPE::U8_C2:
    cv_type = CV_8UC2;
    break;
  case sl::MAT_TYPE::U8_C3:
    cv_type = CV_8UC3;
    break;
  case sl::MAT_TYPE::U8_C4:
    cv_type = CV_8UC4;
    break;
  default:
    break;
  }
  return cv_type;
}

inline cv::Mat slMat_to_cvMat(const sl::Mat &input) {
  // Mapping between MAT_TYPE and CV_TYPE
  return {(int)input.getHeight(), (int)input.getWidth(),
          get_ocv_type(input.getDataType()),
          input.getPtr<sl::uchar1>(sl::MEM::CPU)};
}

enum class MeasurementType {
  ALL,
  IMAGE,
  DEPTH,
  SENSORS,
  OBJECTS,
  IMAGE_AND_SENSORS,
  IMAGE_AND_DEPTH,
  IMAGE_AND_POINT_CLOUD,
  IMAGE_AND_OBJECTS,
};

typedef struct {
  sl::Timestamp timestamp;
  sl::Mat left_image;
  sl::Pose camera_pose;
  sl::Mat depth_map;
  sl::Mat point_cloud;
  sl::SensorsData sensor_data;
  sl::SensorsData::IMUData imu_data;
} ZedMeasurements;

struct zed_config {
  sl::RESOLUTION res = sl::RESOLUTION::AUTO; // Defualt between cameras
  int fps = 0;                               // use max unless specified.
  int flip_camera = sl::FLIP_MODE::AUTO;
  sl::DEPTH_MODE depth_mode = sl::DEPTH_MODE::ULTRA;
  bool sdk_verbose = false;
  sl::COORDINATE_SYSTEM coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
  sl::UNIT units = sl::UNIT::METER;
  float max_depth = 10.0f;

  sl::REFERENCE_FRAME reference_frame = sl::REFERENCE_FRAME::CAMERA;

  bool enable_tracking = true;
  float prediction_timeout_s = 0.0f;
  bool enable_segmentation = false;
  sl::OBJECT_DETECTION_MODEL model =
      sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;

  float detection_confidence_threshold = 0.5f;

  bool enable_batch = false;
  float id_retention_time = 0.0f;
  float batch_latency = 0.0f;

  sl::MEM default_memory = sl::MEM::CPU;

  zed_config() = default;
};

class ZedCamera {
public:
  explicit ZedCamera();
  ZedCamera(const std::string &svo_path);
  ~ZedCamera();

  void configure(const zed_config &config);

  const CAMERA_TYPE get_camera_type() const { return ZED; };

  int open();
  void fetch_measurements(const MeasurementType &type);

  ZedCamera(const ZedCamera &) = delete;
  ZedCamera &operator=(const ZedCamera &) = delete;

  int enable_tracking();
  int enable_object_detection();

  sl::Mat get_depth_map() const;
  const sl::Mat &get_point_cloud() const;
  const sl::Mat &get_left_image() const;
  void get_left_image(sl::Mat &img) const;
  sl::Pose get_camera_pose() const;
  sl::Timestamp get_measurement_timestamp() const;
  sl::SensorsData::IMUData get_imu_data() const;
  void ingest_custom_objects(std::vector<sl::CustomBoxObjectData> &objs);
  const sl::Objects &retrieve_objects() const;
  void set_memory_type(const sl::MEM &memory);
  const sl::ERROR_CODE get_grab_state();

  const sl::Resolution get_resolution() const;
  const sl::Resolution get_svo_resolution();
  CUstream_st *get_cuda_stream();
  void synchronize_cuda_stream();
  void close();

private:
  sl::ERROR_CODE grab_state_;
  sl::Camera zed_;
  sl::InitParameters init_params_;
  sl::RuntimeParameters runtime_params_;
  sl::ObjectDetectionParameters obj_detection_params_;
  sl::ObjectDetectionRuntimeParameters obj_rt_params_;
  sl::BatchParameters batch_params_;
  sl::CalibrationParameters calibration_params_;
  ZedMeasurements measurements_;

  sl::Objects detected_objects_;

  sl::MEM memory_type_;
  bool successful_grab();
  std::string svo_;
};

#endif /* VISION_SYSTEM_ZED_HPP */
#endif /* WITH CUDA */
