#include "vision/apriltag_manager.hpp"

template <typename T>
AprilTagManager<T>::AprilTagManager(const detector_config &cfg) :
        zed_detector_(cfg), monocular_detector_(cfg){
}

template <typename T>
void AprilTagManager<T>::detect_tags(std::vector<tracked_target_info> *targets, const sl::Mat &img, const sl::Mat &point_cloud) {
    apriltag_detection_t *det;
    auto start = std::chrono::high_resolution_clock::now();

    zed_detector_.fetch_detections(slMat_to_cvMat(img));
	if (zed_detector_.has_targets()) {
        targets->clear();
//        info("Zed: " + std::to_string(zed_detector_.get_current_number_of_targets()));
        for (int i = 0; i < zed_detector_.get_current_number_of_targets(); i++) {
            zarray_get(zed_detector_.get_current_detections(), i, &det);
            Corners c = zed_detector_.get_detection_corners(det);
            sl::Vector3<T> tr = get_position_from_pixel(c.tr, point_cloud);
            sl::Vector3<T> tl = get_position_from_pixel(c.tl, point_cloud);
            sl::Vector3<T> br = get_position_from_pixel(c.br, point_cloud);
            if (is_vec_nan(tr) || is_vec_nan(tl) || is_vec_nan(br)){
//              error("Vec is nan");
            } else {
                sl::Pose pose = zed_detector_.get_estimated_target_pose(tr, tl, br);
                error("Found tag: " + std::to_string(det->id));
                // + 1 so that ID 1 will be 2 to not interfere with cube detection id.
                targets->emplace_back(tracked_target_info(pose, det->id + 1));
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        zed_dt_ = duration.count();
//        print_zed_dt();
	}
}

template <typename T>
void AprilTagManager<T>::detect_tags(Zed *camera, std::vector<tracked_target_info> *targets) {
    detect_tags(targets, camera->get_left_image(), camera->get_point_cloud());
}

template <typename T>
void AprilTagManager<T>::detect_tags_monocular(MonocularCamera<T>* camera) {
    std::vector<tracked_target_info> targets;
    apriltag_detection_t *det;
    auto start = std::chrono::high_resolution_clock::now();
    camera->fetch_measurements();

    monocular_detector_.fetch_detections(camera->get_frame());

    if (monocular_detector_.has_targets()) {
//        info("Monocular: " + std::to_string(monocular_detector_.get_current_number_of_targets()));
        for (int i = 0; i < monocular_detector_.get_current_number_of_targets(); i++) {
            zarray_get(monocular_detector_.get_current_detections(), i, &det);
            apriltag_pose_t pose = monocular_detector_.get_estimated_target_pose(
                    camera->get_intrinsic_parameters(),
                    det,
                    (T)0.127 // 5 inch in meters
            );
            targets.emplace_back(
                    tracked_target_info(
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
//        print_monocular_dt();
    }
}

template <typename T>
void AprilTagManager<T>::print_zed_dt() const {
    info("Zed took "  + std::to_string(zed_dt_) + " ms");
}

template <typename T>
void AprilTagManager<T>::print_monocular_dt() const {
    info("Monocular took " + std::to_string(monocular_dt_) + " ms");
}

template <typename T>
AprilTagManager<T>::~AprilTagManager() {}

template class AprilTagManager<float>;
