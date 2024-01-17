#ifdef WITH_CUDA
#include "vision/detections_playback.hpp"
#include "utils/logger.hpp"
#include <chrono>

DetectionsPlayback::DetectionsPlayback(const std::string& svo_file) :
//    yolo_("yolov8s.engine", det_cfg) {
    zed_(svo_file), yolo_("yolov8s.engine") {
    
    yolo_.configure(det_cfg);
    cfg.id_retention_time = 0;
    cfg.prediction_timeout_s = 1;
    zed_.configure(cfg);
    zed_.open_camera();
    zed_.enable_tracking();
    display_resolution = zed_.get_resolution();
    zed_.enable_object_detection();
    video_writer.open("output_video.avi", cv::CAP_FFMPEG, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30, cv::Size(1280, 720), true);
}

DetectionsPlayback::~DetectionsPlayback() {
    zed_.close();
}
    
void DetectionsPlayback::detect() {
    bool running_ = true;
    zed_.set_memory_type(sl::MEM::GPU);
    auto start = std::chrono::high_resolution_clock::now();
    while (running_) {
        zed_.fetch_measurements(MeasurementType::IMAGE);
        //zed_.fetch_measurements();
        auto ret = zed_.get_grab_state();
        auto detections = yolo_.predict(zed_.get_left_image());
        auto err = left_sl.setFrom(zed_.get_left_image(), COPY_TYPE::GPU_CPU);
        
        if (err == sl::ERROR_CODE::SUCCESS) {
            left_cv = slMat_to_cvMat(left_sl);
        } else {
            LOG_ERROR("Failed to update CPU from GPU: ", err);
        }
        cv::imwrite("left_cv.png", left_cv);

        std::vector<sl::CustomBoxObjectData> objects_in;
        for (auto &it : detections) {
            sl::CustomBoxObjectData tmp;
            tmp.unique_object_id = sl::generate_unique_id();
            tmp.probability = it.probability;
            tmp.label = (int) it.label;
            tmp.bounding_box_2d = cvt(it.box);
            tmp.is_grounded = ((int) it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
            objects_in.push_back(tmp);
        }
        zed_.ingest_custom_objects(objects_in);
        zed_.retrieve_objects(objects);
        LOG_DEBUG("Zed objects detected: ", objects.object_list.size());

        for (size_t j = 0; j < objects.object_list.size(); j++) {
            cv::Rect r = sl_cvt_rect(objects.object_list[j].bounding_box_2d);
            cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(left_cv, std::to_string((int) objects.object_list[j].raw_label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            sl::float3 position = objects.object_list[j].position; // Access the XYZ coordinates
            std::string xyz_text = "XYZ: " + std::to_string(position.x) + ", " 
                + std::to_string(position.y) + ", " 
                + std::to_string(position.z);
            cv::putText(left_cv, xyz_text, 
                    cv::Point(r.x, r.y + r.height + 15), // Positioning the text below the bounding box
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0x00, 0xFF, 0xFF), 2);
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
                LOG_ERROR("Video writer not open!");
            }
        } else {
            LOG_ERROR("Frame empty");
        }
        if (ret == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            export_video();
            running_ = false;
        }
    } 
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(stop - start).count();
    LOG_INFO("Detections playback took: ",  elapsed, " seconds");
}
void DetectionsPlayback::export_video() {
    if (video_writer.isOpened()) {
        LOG_INFO("Released writer");
        video_writer.release();
    }
}
#endif /* WITH_CUDA */
