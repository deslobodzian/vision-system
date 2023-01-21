//
// Created by DSlobodzian on 1/2/2022.
//
#include "vision_container.hpp"
#include <iostream>
#include <csignal>

void signal_callback_handler(int signum) {
    std::cout << "Caught signal " << signum << std::endl;
    // Terminate program
    exit(signum);
}
int main() {
    VisionContainer container;
//
    container.run();
    signal(SIGINT, signal_callback_handler);
    return 0;
}

