//
// Created by deslobodzian on 11/21/22.
//

#include "utils/task.hpp"

#ifdef __linux__
#include <sys/timerfd.h>
#include <unistd.h>
#endif

#include <cmath>
#include <stdexcept>
#include <utility>
#include <iostream>

#include "utils/timer.h"

Task::Task(std::shared_ptr<TaskManager> manager, float period, const std::string& name)
    : manager_(manager), period_(period), name_(name) {
}

void Task::start() {
    if (running_.exchange(true)) {
        return; 
    }
    thread_ = std::thread(&Task::loop, this);
}

void Task::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Task::loop() {
    Timer timer;
    init(); // moving here such that in the case the thread restarts the initialization will occur again.
#ifdef __linux__
    auto timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
    if (timer_fd == -1) {
        throw std::runtime_error("Failed to create timerfd");
    }
    long seconds = static_cast<long>(period_);
    long nano_seconds = (static_cast<long>((period_ - sec) * 1e9))

    itimerspec timer_spec{};
    timer_spec.it_interval.tv_sec = seconds;
    timer_spec.it_interval.tv_nsec = nano_seconds;
    timer_spec.it_value = timer_spec.it_inverval;

    if (timerfd_settime(timer_fd, 0, &timer_spec, nullptr) == -1) { 
        close(timer_fd);
        throw std::runtime_error("Failed to set timerfd time");
    }

    uint64_t missed = 0;

    while (running_) {
        timer.start();
        run();
        ssize_t ret = read(timer_fd, &missed, sizeof(missed));
        (void)ret;
    }
#else
    auto ns_period = std::chrono::nanoseconds(static_cast<int>(period_ * 1e9));
    auto next_run = std::chrono::high_resolution_clock::now() + ns_period;

    while (running_) {
        auto start_time = std::chrono::high_resolution_clock::now(); 
        timer.start();
        run();
        auto run_duration = timer.get_nanoseconds();

        next_run = start_time + ns_period;

        auto current_time = std::chrono::high_resolution_clock::now();
        if (current_time < next_run) {
            std::this_thread::sleep_until(next_run);
        }
        std::cout << "Run duration: " << run_duration << " ns" << std::endl;

        auto actual_start = std::chrono::high_resolution_clock::now();

        if (run_duration > ns_period.count()) {
            std::cerr << "Overrun in task " << name_ << ": " << run_duration - ns_period.count() << " ns\n";
        }
    }
#endif
}
Task::~Task() {
    stop();
}

TaskManager::~TaskManager() {
    stop_tasks();
}

void TaskManager::add_task(std::shared_ptr<Task> task) {
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.push_back(task);
}

void TaskManager::stop_tasks() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &task : tasks_) {
        task->stop();
    }
}