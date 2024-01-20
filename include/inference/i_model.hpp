#ifndef VISION_SYSTEM_I_MODEL_HPP
#define VISION_SYSTEM_I_MODEL_HPP

#include "bbox.hpp"
#include "tensor.hpp"

struct detection_config {
  float nms_thres = 0.5;
  float obj_thres = 0.5;
  detection_config() = default;
};

template <typename MatType>
class IModel {
public:
    virtual void configure(const detection_config& cfg) = 0;
    virtual void preprocess(const MatType& image) = 0;
    virtual std::vector<BBoxInfo> postprocess(
        const Tensor<float>& prediction_tensor, 
        const MatType &image) = 0;
    virtual std::vector<BBoxInfo> predict(const MatType& image) = 0;
private:
};
#endif /* VISION_SYSTEM_I_MODEL_HPP */
