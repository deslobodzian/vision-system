#pragma once

#include "task.h"

class SystemContainer{
public:
    SystemContainer();
    ~SystemContainer();
    // void init();
    void run();

    void list_current_tasks();
protected:
    void dummy_thread();
    std::shared_ptr<TaskManager> task_manager_;
};
