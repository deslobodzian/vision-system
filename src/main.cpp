//
// Created by DSlobodzian on 1/2/2022.
//
#include "vision_container.hpp"
#include <cstdlib>
#include <iostream>
#include <csignal>
#include <opencv2/imgcodecs.hpp>
#include "inference/tensor_rt_engine.hpp"
#include "vision/detections_playback.hpp"
#include "vision/apriltag_detector.hpp"

enum MODE {
    BUILD_ENGINE,
    PLAYBACK,
    REALTIME,
    TAG_DETECTION,
    INVALID
};

void signal_callback_handler(int signum) {
    std::cout << "Caught signal " << signum << std::endl;
    // Terminate program
    exit(signum);
}

MODE args_interpreter(int argc, char **argv) {
    if (argc < 2) {
        return REALTIME;
    }
    if (std::string(argv[1]) == "-s" && argc >= 4) {
        // Building engine
#ifdef WITH_CUDA
        std::string onnx_path = argv[2];
        std::string engine_path = argv[3];
        OptimDim dyn_dim_profile;

        EngineConfig cfg;
        cfg.presicion = ModelPrecision::FP_16;
        cfg.int8_data_path = "/home/odin/Data/val2017";
        cfg.onnx_path = onnx_path;
        cfg.engine_path = engine_path;
        cfg.max_threads = 4;
        cfg.optimization_level = 5; // max
        if (argc == 5) {
            std::string optim_profile = argv[4];
            if (dyn_dim_profile.setFromString(optim_profile)) {
                TensorRTEngine::build_engine(cfg, dyn_dim_profile);
                return BUILD_ENGINE;
            } else {
                LOG_ERROR("Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'");
                return INVALID;
            }
            return INVALID;
        }
        TensorRTEngine::build_engine(cfg, dyn_dim_profile);
        return BUILD_ENGINE;
#else
        LOG_ERROR("Cuda not available on this device, cannot build engine!");
        return INVALID;
#endif
    } else if (std::string(argv[1]) == "-p" && argc >= 3) {
#ifdef WITH_CUDA
        std::string playback_file = argv[2];
        DetectionsPlayback playback(playback_file);
        playback.detect();
#else
        LOG_ERROR("Only Zed Camera with Cuda currently implemented");
#endif
        return PLAYBACK;
    } else if (std::string(argv[1]) == "-t" && argc >= 3) {
#ifdef WITH_CUDA
        std::string image_path = argv[2]; 
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error loading the image\n";
            return INVALID;
        }
        LOG_INFO("Tag detection mode, initializing...");
        uint32_t img_width = image.cols;
        uint32_t img_height = image.rows;
        uint32_t tile_size = 4; 
        cuAprilTagsFamily tag_family = NVAT_TAG36H11; 
        float tag_dim = 0.16f; 


        ApriltagDetector detector(img_width, img_height, tile_size, tag_family, tag_dim);
        auto detectedTags = detector.detectAprilTagsInCvImage(image);

        if (image.channels() == 1) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        }

        for (const auto& tag : detectedTags) {
            // Draw tag outlines and IDs
            LOG_INFO("Tag ID: ", tag.id);
            for (int i = 0; i < 4; ++i) {
                LOG_INFO("Corner ", i, ",: {", tag.corners[i].x, ", ", tag.corners[i].y, "}");
                cv::line(image, cv::Point(tag.corners[i].x, tag.corners[i].y),
                        cv::Point(tag.corners[(i + 1) % 4].x, tag.corners[(i + 1) % 4].y),
                        cv::Scalar(0, 255, 0), 2);
            }
            cv::putText(image, std::to_string(tag.id),
                    cv::Point(tag.corners[0].x, tag.corners[0].y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        };

        cv::imwrite("tag_image.jpg", image);

        return TAG_DETECTION;
#else
        LOG_ERROR("CUDA not available on this device, cannot perform tag detection!");
        return INVALID;
#endif
    }
    return REALTIME;
}

int main(int argc, char** argv) {
    MODE mode = args_interpreter(argc, argv);

    switch (mode) {
        case BUILD_ENGINE:
            LOG_INFO("Engine built, exiting program"); 
            return EXIT_SUCCESS;
        case PLAYBACK:
            LOG_INFO("Playback mode, exiting program");
            return EXIT_SUCCESS;
        case REALTIME:
            LOG_INFO("Real-time mode, continuing program");
            break;
        case TAG_DETECTION:
            LOG_INFO("AprilTag Detection mode, exiting program");
            return EXIT_SUCCESS;
        case INVALID:
            break;
        default:
            LOG_ERROR("Invalid arguments or unsupported mode, exiting program");
            return EXIT_FAILURE;
    }
    
    VisionContainer container;
    container.run();
    signal(SIGINT, signal_callback_handler); 
    return EXIT_SUCCESS;
}

