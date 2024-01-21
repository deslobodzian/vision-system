#ifndef VISION_SYSTEM_OBJECT_DETECTOR_HPP
#define VISION_SYSTEM_OBJECT_DETECTOR_HPP

#include "inference/i_model.hpp"
#include "inference/yolo.hpp"
#include "vision/monocular_camera.hpp"

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
    ObjectDetector(const detection_config& cfg) {
        config_ = cfg;
        model_ = std::make_unique<Yolo<MatType>>("yolov8s.onnx");
        model_->configure(config_);
    }
    ~ObjectDetector() {
    }
    void detect_objects(const Camera& camera) {
        detect_objects_impl(camera);
    }
private:
    detection_config config_;
    std::unique_ptr<IModel<MatType>> model_;

#ifdef WITH_CUDA
    void detect_objects_impl(const ZedCamera& camera);
#endif
    void detect_objects_impl(const MonocularCamera& camera) {
        cv::Mat img = camera.get_frame();
        std::vector<BBoxInfo> obj =  model_->predict(img);
        LOG_DEBUG("Detected ", obj.size(), " objects");
    }
};



#endif /* VISION_SYSTEM_OBJECT_DETECTOR_HPP */
