#include "task.h"

#include <chrono>
#include <memory>
#include <catch2/catch_test_macros.hpp>

class TestTask : public Task {
 public:
  TestTask(std::shared_ptr<TaskManager> manager, float period, std::string name)
      : Task(manager, period, name), run_count(0) {}

  void init() override {}

  void run() override { ++run_count; }

  int run_count;
};

TEST_CASE("TaskTest - Period", "[task][periodicity]") {
  auto manager = std::make_shared<TaskManager>();
  auto task = manager->create_task<TestTask>(0.1f, "TestTask");

  task->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  task->stop();

    REQUIRE(task->run_count >= 4);
    REQUIRE(task->run_count <= 6);
}

TEST_CASE("TaskTest - MultiplePeriods", "[task][multiple_periods]") {
  auto manager = std::make_shared<TaskManager>();
  auto task1 = manager->create_task<TestTask>(0.1f, "TestTask1");
  auto task2 = manager->create_task<TestTask>(0.2f, "TestTask2");

  task1->start();
  task2->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  task1->stop();
  task2->stop();

    REQUIRE(task1->run_count >= 9);
    REQUIRE(task1->run_count <= 11);
    REQUIRE(task2->run_count >= 4);
    REQUIRE(task2->run_count <= 6);
}

TEST_CASE("TaskTest - StartStop", "[task][start_stop]") {
  auto manager = std::make_shared<TaskManager>();
  auto task = manager->create_task<TestTask>(0.1f, "TestTask");

  task->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  task->stop();
  int run_count_after_stop = task->run_count;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
    REQUIRE(task->run_count == run_count_after_stop);
}

TEST_CASE("TaskManagerTest - Stop", "[task_manager][stop]") {
  auto manager = std::make_shared<TaskManager>();
  auto task1 = manager->create_task<TestTask>(0.1f, "TestTask1");
  auto task2 = manager->create_task<TestTask>(0.2f, "TestTask2");

  task1->start();
  task2->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  manager->stop_tasks();
  int run_count1_after_stop = task1->run_count;
  int run_count2_after_stop = task2->run_count;
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(task1->run_count == run_count1_after_stop);
    REQUIRE(task2->run_count == run_count2_after_stop);
}
