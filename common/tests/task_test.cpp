#include "task.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

class TestTask : public Task {
 public:
  TestTask(std::shared_ptr<TaskManager> manager, float period, std::string name)
      : Task(manager, period, name), run_count(0) {}

  void init() override {}

  void run() override { ++run_count; }

  int run_count;
};

TEST(TaskTest, Periodicity) {
  auto manager = std::make_shared<TaskManager>();
  auto task = manager->create_task<TestTask>(0.1f, "TestTask");

  task->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  task->stop();

  EXPECT_GE(task->run_count, 4);
  EXPECT_LE(task->run_count, 6);
}

TEST(TaskTest, MultipleTasksWithDifferentPeriods) {
  auto manager = std::make_shared<TaskManager>();
  auto task1 = manager->create_task<TestTask>(0.1f, "TestTask1");
  auto task2 = manager->create_task<TestTask>(0.2f, "TestTask2");

  task1->start();
  task2->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  task1->stop();
  task2->stop();

  EXPECT_GE(task1->run_count, 9);
  EXPECT_LE(task1->run_count, 11);
  EXPECT_GE(task2->run_count, 4);
  EXPECT_LE(task2->run_count, 6);
}

TEST(TaskTest, StartStop) {
  auto manager = std::make_shared<TaskManager>();
  auto task = manager->create_task<TestTask>(0.1f, "TestTask");

  task->start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  task->stop();
  int run_count_after_stop = task->run_count;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(task->run_count, run_count_after_stop);
}

TEST(TaskManagerTest, StopTasks) {
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
  EXPECT_EQ(task1->run_count, run_count1_after_stop);
  EXPECT_EQ(task2->run_count, run_count2_after_stop);
}
