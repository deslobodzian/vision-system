#ifndef VISION_SYSTEM_OBJECT_DETECTOR_HPP
#define VISION_SYSTEM_OBJECT_DETECTOR_HPP

#include "inference/i_model.hpp"
#include "inference/yolo.hpp"
#include "vision/monocular_camera.hpp"

#ifdef WITH_CUDA
#include "detection_utils.hpp"
#include "zed.hpp"
#endif

template <typename Camera>
struct CameraTraits;

#ifdef WITH_CUDA
template <>
struct CameraTraits<ZedCamera> {
  using MatType = sl::Mat;
};
#endif

template <>
struct CameraTraits<MonocularCamera> {
  using MatType = cv::Mat;
};

template <typename Camera>
class ObjectDetector {
 public:
  using MatType = typename CameraTraits<Camera>::MatType;
  ObjectDetector() {
    // model_ = std::make_unique<Yolo<MatType>>("coco");
    model_ = std::make_unique<Yolo<MatType>>("note");
    detections_.reserve(
        100);  // There should never be more that 100 object detected at once
  }
  ~ObjectDetector() {}

  void configure(const detection_config& config) { model_->configure(config); }

  void detect_objects(Camera& camera) { detect_objects_impl(camera); }

 private:
  detection_config config_;
  std::unique_ptr<IModel<MatType>> model_;
  std::vector<BBoxInfo> detections_;

#ifdef WITH_CUDA
  void detect_objects_impl(ZedCamera& camera) {
    camera.synchronize_cuda_stream();
    auto new_detections = model_->predict(camera.get_left_image());
    detections_.swap(new_detections);
    LOG_DEBUG("Model detections: ", detections_.size());
    std::vector<sl::CustomBoxObjectData> objects_in;
    for (auto& it : detections_) {
      sl::CustomBoxObjectData tmp;
      tmp.unique_object_id = sl::generate_unique_id();
      tmp.probability = it.probability;
      tmp.label = (int)it.label;
      tmp.bounding_box_2d = cvt(it.box);
      tmp.is_grounded =
          ((int)it.label == 0);  // Only the first class (person) is grounded,
                                 // that is moving on the floor plane
      objects_in.push_back(tmp);
    }
    camera.ingest_custom_objects(objects_in);
    // const sl::Objects &obj = camera.retrieve_objects();
    // LOG_DEBUG("Zed objects detected: ", obj.object_list.size());
  }
#endif
  void detect_objects_impl(MonocularCamera& camera) {
    cv::Mat img = camera.get_frame();
    std::vector<BBoxInfo> obj = model_->predict(img);
    LOG_DEBUG("Detected ", obj.size(), " objects");
  }
};

#endif /* VISION_SYSTEM_OBJECT_DETECTOR_HPP */
