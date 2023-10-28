#include "vision/detections_playback.hpp"
DetectionsPlayback::DetectionsPlayback(std::string svo_file) {
    sl::InitParameters init_params;
    init_params.sdk_verbose = true;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_params.input.setFromSVOFile(svo_file.c_str());
    auto ret = zed_.open(init_params);
    if (ret != sl::ERROR_CODE::SUCCESS) {
        error("Camera failed to open in playback");
    }

    zed_.enablePositionalTracking();
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = false; // designed to give person pixel mask
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    ret = zed_.enableObjectDetection(detection_parameters);
    if (ret != sl::ERROR_CODE::SUCCESS) {
        zed_.close();
    }
    auto camera_config = zed_.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed_.getCameraInformation(pc_resolution).camera_configuration;
    display_resolution = zed_.getCameraInformation().camera_configuration.resolution;

    video_writer.open("output_video.avi", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(1280, 720), true);
   
    yolo_.init("yolov8s.engine");
}
DetectionsPlayback::~DetectionsPlayback() {
    zed_.close();
}
    
void DetectionsPlayback::detect() {
    bool running_ = true;

    while (running_) {
        auto ret = zed_.grab();
        zed_.retrieveImage(left_sl, sl::VIEW::LEFT, sl::MEM::GPU);
        // zed_.retrieveImage(left_sl, sl::VIEW::LEFT, sl::MEM::CPU);
        auto detections = yolo_.run(left_sl, display_resolution.height, display_resolution.width, CONF_THRESH);

        if (left_sl.updateCPUfromGPU() == sl::ERROR_CODE::SUCCESS) {
            left_cv = slMat_to_cvMat(left_sl);
        }

        std::vector<sl::CustomBoxObjectData> objects_in;
        for (auto &it : detections) {
            sl::CustomBoxObjectData tmp;
            tmp.unique_object_id = sl::generate_unique_id();
            tmp.probability = it.prob;
            tmp.label = (int) it.label;
            tmp.bounding_box_2d = cvt(it.box);
            tmp.is_grounded = ((int) it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
            objects_in.push_back(tmp);
        }
        zed_.ingestCustomBoxObjects(objects_in);

        for (size_t j = 0; j < detections.size(); j++) {
            cv::Rect r = cvt_rect(detections[j].box);
            cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(left_cv, std::to_string((int) detections[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        if (left_cv.type() == CV_8UC3) {
        } else if (left_cv.type() == CV_8UC4) {
            cv::Mat threeChannelMat;
            cv::cvtColor(left_cv, threeChannelMat, cv::COLOR_BGRA2BGR);
            left_cv = threeChannelMat; 
        } else {
            cv::cvtColor(left_cv, left_cv, cv::COLOR_GRAY2BGR); 
        }
        if (!left_cv.empty()) {
            if (video_writer.isOpened()) {
                video_writer.write(left_cv);
            } else {
                error("Video writer not open!");
            }
        } else {
            error("Frame empty");
        }
        if (ret == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            export_video();
            running_ = false;
        }
    } 
}
void DetectionsPlayback::export_video() {
    if (video_writer.isOpened()) {
        info("Released writer");
        video_writer.release();
    }
}