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

extern "C" {
	#include "apriltag.h"
	#include "tag36h11.h"
	#include "common/getopt.h"
}

class AprilTagManager {

private:
	std::vector<std::thread> threads_;

public:
	void add_detector_thread(Zed & camera);
	void add_detector_thread(MonocularCamera & camera);
	void detector_monocular(MonocularCamera & camera);
	void detector_zed(Zed & zed);

};



