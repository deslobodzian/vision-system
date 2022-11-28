#pragma once

/*
	Manager will track the current located AprilTag detected by each device 

	Zed camera will provide depth and bearing from apriltag by overlaying the depth map with the pixel coordinates

		* Compare the difference between apriltags solve pnp and sterolabs depth map. 

	Mononcular camera will also provid depth from AprilTags SolvePNP and bearing. 


	Manager will poll data from each device in seperate threads 

	Vector with locate targets using tracked_object_info objects.

*/

#include <vector>
#include <thread>
#include "vision/Zed.hpp"
#include "vision/monocular_camera.hpp"
#include <opencv2/opencv.hpp>
#include "apriltag_detector.hpp"
#include "utils/utils.hpp"
#include "tracked_target_info.hpp"

extern "C" {
	#include "apriltag.h"
	#include "tag36h11.h"
	#include "common/getopt.h"
}

class AprilTagManager {
public:
    AprilTagManager() = default;
    explicit AprilTagManager(const detector_config &cfg);
    ~AprilTagManager();

    template <typename T>
	void detect_tags_monocular(MonocularCamera<T>* camera) {
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
	void detect_tags_zed(Zed* zed);
    void print_monocular_dt() const;
    void print_zed_dt() const;

    std::vector<TrackedTargetInfo> get_zed_targets();
    std::vector<TrackedTargetInfo> get_monocular_targets();
private:
    TagDetector zed_detector_;
    TagDetector monocular_detector_;
    std::mutex zed_mtx_;
    std::mutex monocular_mtx_;
    std::vector<TrackedTargetInfo> zed_targets_;
    std::vector<TrackedTargetInfo> monocular_targets_;
    long zed_dt_{};
    long monocular_dt_{};

};



