#include "vision/apriltag_manager.hpp"

template <typename T>
AprilTagManager<T>::AprilTagManager(const detector_config &cfg) :
        zed_detector_(cfg), monocular_detector_(cfg){
}

template <typename T>
void AprilTagManager<T>::detect_tags_zed(Zed* camera) {
    std::vector<TrackedTargetInfo> targets;
    apriltag_detection_t *det;
    auto start = std::chrono::high_resolution_clock::now();
    camera->fetch_measurements();
    zed_detector_.fetch_detections(slMat_to_cvMat(camera->get_left_image()));

	if (zed_detector_.has_targets()) {
        targets.clear();
        info("Zed: " + std::to_string(zed_detector_.get_current_number_of_targets()));
        for (int i = 0; i < zed_detector_.get_current_number_of_targets(); i++) {
            zarray_get(zed_detector_.get_current_detections(), i, &det);
            Corners c = zed_detector_.get_detection_corners(det);
            sl::Vector3<T> tr = camera->get_position_from_pixel(c.tr);
            sl::Vector3<T> tl = camera->get_position_from_pixel(c.tl);
            sl::Vector3<T> br = camera->get_position_from_pixel(c.br);
            if (is_vec_nan(tr) || is_vec_nan(tl) || is_vec_nan(br)){
//              error("Vec is nan");
            } else {
                sl::Pose pose = zed_detector_.get_estimated_target_pose(tr, tl, br);
                targets.emplace_back(TrackedTargetInfo(pose, det->id));
            }
        }
        const std::lock_guard<std::mutex> lock(zed_mtx_);
        zed_targets_ = targets;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        zed_dt_ = duration.count();
	}
}

template <typename T>
void AprilTagManager<T>::detect_tags_monocular(MonocularCamera<T>* camera) {
    std::vector<TrackedTargetInfo> targets;
    apriltag_detection_t *det;
    auto start = std::chrono::high_resolution_clock::now();
    camera->fetch_measurements();
    monocular_detector_.fetch_detections(camera->get_frame());

    if (monocular_detector_.has_targets()) {
        info("Monocular: " + std::to_string(monocular_detector_.get_current_number_of_targets()));
        for (int i = 0; i < monocular_detector_.get_current_number_of_targets(); i++) {
            zarray_get(monocular_detector_.get_current_detections(), i, &det);
            apriltag_pose_t pose = monocular_detector_.get_estimated_target_pose(
                    camera->get_intrinsic_parameters(),
                    det,
                    (T)0.127 // 5 inch in meters
            );
            targets.emplace_back(
                    TrackedTargetInfo(
                            pose.t->data[0],
                            pose.t->data[1],
                            pose.t->data[2],
                            det->id)
            );
        }
        const std::lock_guard<std::mutex> lock(monocular_mtx_);
        monocular_targets_ = targets;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        monocular_dt_ = duration.count();
    }
}

template <typename T>
std::vector<TrackedTargetInfo> AprilTagManager<T>::get_zed_targets() {
    const std::lock_guard<std::mutex> lock(zed_mtx_);
    return zed_targets_;
}

template <typename T>
std::vector<TrackedTargetInfo> AprilTagManager<T>::get_monocular_targets() {
    const std::lock_guard<std::mutex> lock(monocular_mtx_);
    return monocular_targets_;
}

template <typename T>
void AprilTagManager<T>::print_zed_dt() const {
    info("Zed took "  + std::to_string(zed_dt_));
}

template <typename T>
void AprilTagManager<T>::print_monocular_dt() const {
    info("Monocular took " + std::to_string(monocular_dt_));
}

template <typename T>
AprilTagManager<T>::~AprilTagManager() {}

template class AprilTagManager<float>;
