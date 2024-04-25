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

inline sl::float3 normalize(const sl::float3 &v) {
  float norm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return {v.x / norm, v.y / norm, v.z / norm};
}

inline cv::Mat slMat_to_cvMat(const sl::Mat &input) {
  // Mapping between MAT_TYPE and CV_TYPE
  return {(int)input.getHeight(), (int)input.getWidth(),
          get_ocv_type(input.getDataType()),
          input.getPtr<sl::uchar1>(sl::MEM::CPU)};
}

inline sl::Orientation
compute_orientation_from_normal(const sl::float3 &normal) {
  sl::float3 up_vector = {0.0f, 0.0f, 1.0f};

  sl::float3 normalized_normal = normalize(normal);

  sl::float3 rotation_axis = {
      up_vector.y * normalized_normal.z - up_vector.z * normalized_normal.y,
      up_vector.z * normalized_normal.x - up_vector.x * normalized_normal.z,
      up_vector.x * normalized_normal.y - up_vector.y * normalized_normal.x};

  rotation_axis = normalize(rotation_axis);

  float dot_product = sl::float3::dot(up_vector, normalized_normal);
  // float dot_product = up_vector.x * normalized_normal.x + up_vector.y *
  // normalized_normal.y + up_vector.z * normalized_normal.z;
  float angle = acos(dot_product); // Angle in radians

  float s = sin(angle / 2);
  sl::Orientation orientation;
  orientation[0] = rotation_axis.x * s; // ox
  orientation[1] = rotation_axis.y * s; // oy
  orientation[2] = rotation_axis.z * s; // oz
  orientation[3] = cos(angle / 2);      // ow

  orientation.normalise();

  return orientation;
}

enum class MeasurementType {
  ALL,
  IMAGE,
  DEPTH,
  SENSORS,
  OBJECTS,
  IMAGE_AND_SENSORS,
  IMAGE_AND_DEPTH,
  IMAGE_AND_POINT_CLOUD, // this also gets normals
  IMAGE_AND_OBJECTS,
};

typedef struct {
  sl::Timestamp timestamp;
  sl::Mat left_image;
  sl::Pose camera_pose;
  sl::Mat depth_map;
  sl::Mat point_cloud;
  sl::Mat normals;
  sl::SensorsData sensor_data;
  sl::SensorsData::IMUData imu_data;
} ZedMeasurements;

struct zed_config {
  unsigned int serial_number = 0;
  sl::RESOLUTION res = sl::RESOLUTION::AUTO; // Defualt between cameras
  int fps = 0;                               // use max unless specified.
  int flip_camera = sl::FLIP_MODE::AUTO;
  sl::DEPTH_MODE depth_mode = sl::DEPTH_MODE::ULTRA;
  bool sdk_verbose = false;
  sl::COORDINATE_SYSTEM coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
  sl::UNIT units = sl::UNIT::METER;
  float max_depth = 10.0f;
  float min_depth = 0.3f;
  bool async_grab = false;

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
  sl::Resolution depth_resolution = sl::getResolution(sl::RESOLUTION::AUTO);

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
  const sl::Mat &get_normals() const;
  const sl::Mat &get_left_image() const;
  sl::ERROR_CODE get_plane(const sl::uint2 &point, sl::Plane &plane);
  void get_left_image(sl::Mat &img) const;
  sl::Pose get_camera_pose() const;
  sl::Timestamp get_measurement_timestamp() const;
  sl::SensorsData::IMUData get_imu_data() const;
  void ingest_custom_objects(std::vector<sl::CustomBoxObjectData> &objs);
  void fetch_objects();
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
  sl::Resolution depth_resolution_;
  bool successful_grab();
  std::string svo_;
};

#endif /* VISION_SYSTEM_ZED_HPP */
#endif /* WITH CUDA */
