#include "system_container.h"
#include "logger.h"
#include "task.h"

class TestTask : public Task {
public:
    TestTask(
        std::shared_ptr<TaskManager>,
        float period,
        const std::string&
    );
    void init() override;
    void run() override;
};

TestTask::TestTask(std::shared_ptr<TaskManager> task_manager, float period, const std::string& name) 
    : Task(task_manager, period, name) {
}

void TestTask::init() {

}
void TestTask::run() {
    LOG_INFO("Dummy Test");
}

SystemContainer::SystemContainer() : task_manager_(std::make_shared<TaskManager>()) {

}

SystemContainer::~SystemContainer() {
}

void SystemContainer::dummy_thread() {
    LOG_INFO("Dummy Test");
}

void SystemContainer::run() {
    auto test_task = task_manager_->create_task<TestTask>(
        5.0f, // seconds
        "test"
    );
    test_task->start();
}

void SystemContainer::list_current_tasks() {
    auto tasks = task_manager_->get_current_tasks();

    for (auto task : tasks) {
        LOG_INFO(task->status_string());
    }
}

