#ifndef VISION_SYSTEM_SYSTEM_CONTAINER_H
#define VISION_SYSTEM_SYSTEM_CONTAINER_H

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

#endif /* VISION_SYSTEM_SYSTEM_CONTAINER_H */