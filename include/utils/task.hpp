//
// Created by deslobodzian on 11/21/22.
//

#ifndef VISION_SYSTEM_TASK_HPP
#define VISION_SYSTEM_TASK_HPP

#include <string>
#include <thread>
#include <sys/timerfd.h>
#include <vector>
//#include "task_manager.hpp"
#include <cmath>
/* task that will run at a specific period in hz. */
class TaskManager;

class Task {
public:
//    Task(TaskManager* manager, double period, std::string name);
    Task(TaskManager* manager, float period, std::string name);
    virtual ~Task() {stop();}
    void start();
    void stop();
    virtual void init();
    virtual void run();

    /* return period in hz */
    float get_period() {return period_;}


private:
    double period_;
    bool running_ = false;
    std::string name_;
    std::thread thread_;

    void loop();
};

class TaskManager {
public:
    TaskManager() = default;
    ~TaskManager();
    void add_task(Task* task);
    void stop_tasks();
private:
    std::vector<Task*> tasks_;
};

class PeriodicFunction : public Task{
public:
    PeriodicFunction(TaskManager* taskManager, float period,
                     std::string name, void (*function)())
            : Task(taskManager, period, name), _function(function) {}
    void init() {}
    void run() { _function(); }

    ~PeriodicFunction() { stop(); }

private:
    void (*_function)() = nullptr;
};

template <typename T>
class PeriodicMemberFunction : public Task {
public:
    PeriodicMemberFunction(TaskManager* taskManager, float period,
                           std::string name, void (T::*function)(), T* obj)
            : Task(taskManager, period, name),
              _function(function),
              _obj(obj) {}

    void init() {}
    void run() { (_obj->*_function)(); }

private:
    void (T::*_function)();
    T* _obj;
};
#endif //VISION_SYSTEM_TASK_HPP