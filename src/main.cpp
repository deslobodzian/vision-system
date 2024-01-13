//
// Created by DSlobodzian on 1/2/2022.
//
#include "vision_container.hpp"
#include <iostream>
#include <csignal>
#include "inference/tensor_rt_engine.hpp"
#include "vision/detections_playback.hpp"

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
        return REALTIME;
    }
    if (std::string(argv[1]) == "-s" && argc >= 4) {
        // Building engine
#ifdef WITH_CUDA
         std::string onnx_path = argv[2];
         std::string engine_path = argv[3];
         OptimDim dyn_dim_profile;

        if (argc == 5) {
            std::string optim_profile = argv[4];
             if (dyn_dim_profile.setFromString(optim_profile)) {
                 TensorRTEngine::build_engine(onnx_path, engine_path, dyn_dim_profile);
                 return BUILD_ENGINE;
             } else {
                 LOG_ERROR("Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'");
                 return INVALID;
             }
            return INVALID;
        }
        TensorRTEngine::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return BUILD_ENGINE;
#else
        LOG_ERROR("Cuda not available on this device, cannot build engine!");
        return INVALID;
#endif
    }
    else if (std::string(argv[1]) == "-p" && argc >= 3) {
#ifdef WITH_CUDA
        std::string playback_file = argv[2];
        DetectionsPlayback playback(playback_file);
        playback.detect();
#else
        LOG_ERROR("Only Zed Camera with Cuda currently implemented");
#endif
        return PLAYBACK;
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
        case INVALID:
        default:
            LOG_ERROR("Invalid arguments or unsupported mode, exiting program");
            return EXIT_FAILURE;
    }
    
    VisionContainer container;
    container.run();
    signal(SIGINT, signal_callback_handler); 
    return EXIT_SUCCESS;
}

