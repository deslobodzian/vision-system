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
    AprilTagManager() = default;
    AprilTagManager(const DetectorConfig &cfg);
    ~AprilTagManager();
	void add_detector_thread(Zed* camera);
    template <typename T>
	void add_detector_thread(MonocularCamera<T>* camera);
    template <typename T>
	void detector_monocular(MonocularCamera<T>* camera);
	void detector_zed(Zed* zed);
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
    long zed_dt_;
    long monocular_dt_;
};



