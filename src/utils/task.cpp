//
// Created by deslobodzian on 11/21/22.
//

#include "utils/task.hpp"
#include "utils/task_manager.hpp"

Task::Task(TaskManager* manager, double period, std::string name)
    : period_(period), name_(name) {
    manager->add_task(this);
}

void Task::start() {
    if (running_) {
        return;
    }
    init();
    running_ = true;
    thread_ = std::thread(&Task::loop, this);
}

void Task::stop() {
    if (!running_) {
        return;
    }
    running_ = false;
    thread_.join();
}

void Task::loop() {
    auto timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
    int seconds = (int) period_;
    int nano_seconds = (int)(1e9 * std::fmod(period_, 1.f));

    itimerspec timer_spec{};
    timer_spec.it_interval.tv_sec = seconds;
    timer_spec.it_value.tv_sec = seconds;

    timer_spec.it_interval.tv_nsec = nano_seconds;
    timer_spec.it_value.tv_nsec = nano_seconds;

    timerfd_settime(timer_fd, 0, &timer_spec, nullptr);

    unsigned long long missed = 0;

    while (running_) {
        run();
        int m = read(timer_fd, &missed, sizeof(missed));
        (void)m;
    }
}