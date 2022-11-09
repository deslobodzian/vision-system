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
	#include "tag25h9.h"
	#include "tag16h5.h"
	#include "tagCircle21h7.h"
	#include "tagCircle49h12.h"
	#include "tagCustom48h12.h"
	#include "tagStandard41h12.h"
	#include "tagStandard52h13.h"
	#include "common/getopt.h"
}

class AprilTagManager() {

private:
	std::vector<std::thread> threads_;

public:
	void add_fiducial_thread(Zed & camera);
	void add_fiducial_thread(MonocularCamera & camera);
	void find_fiducials_monocular();
	void find_fiducials_zed();

}



