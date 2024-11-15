//
// Created by deslobodzian on 11/21/22.
//

#ifndef VISION_SYSTEM_TASK_HPP
#define VISION_SYSTEM_TASK_HPP

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

/* task that will run at a specific period in hz. */
class TaskManager;

class Task : public std::enable_shared_from_this<Task> {
 public:
  Task(const Task&) = delete;
  Task(Task&&) = delete;
  auto operator=(const Task&) -> Task& = delete;
  auto operator=(Task&&) -> Task& = delete;
  //    Task(TaskManager* manager, double period, std::string name);
  Task(std::shared_ptr<TaskManager> manager, float period, std::string name);
  virtual ~Task();

  void start();
  void stop();
  virtual void init() = 0;
  virtual void run() = 0;

  auto get_period() const -> float { return 1.0F / period_; }

 private:
  std::shared_ptr<TaskManager> manager_;
  float period_;
  std::atomic<bool> running_{false};
  std::string name_;
  std::thread thread_;

  void loop();
};

class TaskManager : public std::enable_shared_from_this<TaskManager> {
 public:
  TaskManager() = default;
  ~TaskManager();

  void add_task(std::shared_ptr<Task> task);
  template <typename T, typename... Args>
  auto create_task(Args&&... args) -> std::shared_ptr<T> {
    static_assert(std::is_base_of_v<Task, T>, "T must be a derivative of Task");

    auto task = std::make_shared<T>(this->shared_from_this(),
                                    std::forward<Args>(args)...);
    add_task(task);
    return task;
  }
  void stop_tasks();

 private:
  std::vector<std::shared_ptr<Task>> tasks_;
  std::mutex mtx_;
};

class PeriodicFunction : public Task {
 public:
  PeriodicFunction(const PeriodicFunction&) = delete;
  PeriodicFunction(PeriodicFunction&&) = delete;
  auto operator=(const PeriodicFunction&) -> PeriodicFunction& = delete;
  auto operator=(PeriodicFunction&&) -> PeriodicFunction& = delete;
  PeriodicFunction(std::shared_ptr<TaskManager> taskManager, float period,
                   const std::string& name, std::function<void()> function)
      : Task(std::move(taskManager), period, name),
        function_(std::move(function)) {}

  void init() override {}
  void run() override {
    if (function_) {
      function_();
    }
  }

  ~PeriodicFunction() override { stop(); }

 private:
  std::function<void()> function_;
};

template <typename T>
class PeriodicMemberFunction : public Task {
 public:
  PeriodicMemberFunction(std::shared_ptr<TaskManager> taskManager, float period,
                         const std::string& name, T* obj, void (T::*function)())
      : Task(taskManager, period, name), function_(std::bind(function, obj)) {}

  void init() override {}
  void run() override {
    if (function_) {
      function_();
    }
  }

 private:
  std::function<void()> function_;
};

#endif  // VISION_SYSTEM_TASK_HPP
