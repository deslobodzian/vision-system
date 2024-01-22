#ifdef WITH_CUDA

#ifndef VISION_SYSTEM_ZED_HPP
#define VISION_SYSTEM_ZED_HPP

#include "i_camera.hpp"
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <vector>

using namespace sl;

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

enum class MeasurementType { ALL, IMAGE, DEPTH, SENSORS, IMAGE_AND_SENSORS };

typedef struct {
  Timestamp timestamp;
  Mat left_image;
  Pose camera_pose;
  Mat depth_map;
  Mat point_cloud;
  SensorsData sensor_data;
  SensorsData::IMUData imu_data;
} ZedMeasurements;

struct zed_config {
  RESOLUTION res = RESOLUTION::HD1080; // Defualt at 1080p
  int fps = 0;                         // use max unless specified.
  int flip_camera = FLIP_MODE::AUTO;
  DEPTH_MODE depth_mode = DEPTH_MODE::ULTRA;
  bool sdk_verbose = true;
  COORDINATE_SYSTEM coordinate_system = COORDINATE_SYSTEM::IMAGE;
  UNIT units = UNIT::METER;
  float max_depth = 10.0f;

  REFERENCE_FRAME reference_frame = REFERENCE_FRAME::CAMERA;

  bool enable_tracking = true;
  float prediction_timeout_s = 0.0f;
  bool enable_segmentation = false;
  OBJECT_DETECTION_MODEL model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;

  float detection_confidence_threshold = 0.5f;

  bool enable_batch = false;
  float id_retention_time = 0.0f;
  float batch_latency = 0.0f;

  MEM default_memory = MEM::CPU;

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

  Mat get_depth_map() const;
  Mat get_point_cloud() const;
  const Mat& get_left_image() const;
  void get_left_image(Mat &img) const;
  Pose get_camera_pose() const;
  Timestamp get_measurement_timestamp() const;
  SensorsData::IMUData get_imu_data() const;
  void ingest_custom_objects(std::vector<CustomBoxObjectData> &objs);
  const Objects& retrieve_objects() const;
  void set_memory_type(const MEM &memory);
  const ERROR_CODE get_grab_state();

  const Resolution get_resolution() const;
  void close();

private:
  ERROR_CODE grab_state_;
  Camera zed_;
  InitParameters init_params_;
  RuntimeParameters runtime_params_;
  ObjectDetectionParameters obj_detection_params_;
  ObjectDetectionRuntimeParameters obj_rt_params_;
  BatchParameters batch_params_;
  CalibrationParameters calibration_params_;
  ZedMeasurements measurements_;

  Objects detected_objects_;

  MEM memory_type_;
  bool successful_grab();
  std::string svo_;
};

#endif /* VISION_SYSTEM_ZED_HPP */
#endif /* WITH CUDA */
