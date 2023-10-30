//
// Created by DSlobodzian on 1/2/2022.
//
#include "vision_container.hpp"
#include "inference/yolo.hpp"
#include "vision/detections_playback.hpp"
#include <iostream>
#include <csignal>


enum MODE {
    BUILD_ENGINE,
    PLAYBACK,
    REALTIME,
    INVALID
};

void signal_callback_handler(int signum) {
    std::cout << "Caught signal " << signum << std::endl;
    // Terminate program
    exit(signum);
}

MODE args_interpreter(int argc, char **argv) {
    if (argc < 2) {
        info("Running realtime!");
        return REALTIME;
    }
    if (std::string(argv[1]) == "-s" && argc >= 4) {
        // Building engine
        std::string onnx_path = argv[2];
        std::string engine_path = argv[3];
        OptimDim dyn_dim_profile;

        if (argc == 5) {
            std::string optim_profile = argv[4];
            if (dyn_dim_profile.setFromString(optim_profile)) {
                Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
                return BUILD_ENGINE;
            } else {
                std::cerr << "Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'" << std::endl;
                return INVALID;
            }
        }
        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return BUILD_ENGINE;
    }
    else if (std::string(argv[1]) == "-p" && argc >= 3) {
        std::string playback_file = argv[2];
        DetectionsPlayback playback(playback_file);
        playback.detect();
        return PLAYBACK;
    }
    return REALTIME;
}

int main(int argc, char** argv) {
    MODE mode = args_interpreter(argc, argv);

    switch (mode) {
        case BUILD_ENGINE:
            info("Engine built, exiting program"); 
            return EXIT_SUCCESS;
        case PLAYBACK:
            info("Playback mode, exiting program");
            return EXIT_SUCCESS;
        case REALTIME:
            info("Real-time mode, continuing program");
            break;
        case INVALID:
        default:
            error("Invalid arguments or unsupported mode, exiting program");
            return EXIT_FAILURE;
    }
    
    VisionContainer container;
    container.run();
    signal(SIGINT, signal_callback_handler); 
    return EXIT_SUCCESS;
}

