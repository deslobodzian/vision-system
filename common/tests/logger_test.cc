#include "logger.h"

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>

TEST(LoggerTest, LogLevels) {
  logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);

  std::stringstream ss;
  std::streambuf *old_cout_buffer = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  LOG_DEBUG("Debug message");
  LOG_INFO("Info message");
  LOG_ERROR("Error message");

  std::this_thread::sleep_for(
      std::chrono::milliseconds(100));  // Wait for async logging

  std::cout.rdbuf(old_cout_buffer);

  std::string log_output = ss.str();
  EXPECT_TRUE(log_output.find("Debug message") != std::string::npos);
  EXPECT_TRUE(log_output.find("Info message") != std::string::npos);
  EXPECT_TRUE(log_output.find("Error message") != std::string::npos);
}

TEST(LoggerTest, LogFile) {
  logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);
  std::string log_file = "test_log.txt";
  logger::Logger::instance().set_log_file(log_file);

  LOG_DEBUG("Debug message");
  LOG_INFO("Info message");
  LOG_ERROR("Error message");

  std::this_thread::sleep_for(
      std::chrono::milliseconds(100));  // Wait for async logging

  std::ifstream file(log_file);
  std::stringstream ss;
  ss << file.rdbuf();
  std::string log_output = ss.str();

  EXPECT_TRUE(log_output.find("Debug message") != std::string::npos);
  EXPECT_TRUE(log_output.find("Info message") != std::string::npos);
  EXPECT_TRUE(log_output.find("Error message") != std::string::npos);

  std::remove(log_file.c_str());
}

void log_from_thread(int thread_id) {
  LOG_DEBUG("Debug message from thread ", thread_id);
  LOG_INFO("Info message from thread ", thread_id);
  LOG_ERROR("Error message from thread ", thread_id);
}

TEST(LoggerTest, MultiThreadedLogging) {
  logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);

  std::string log_file = "test_log_multithread.txt";
  logger::Logger::instance().set_log_file(log_file);

  const int num_threads = 5;
  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(log_from_thread, i);
  }

  for (auto &thread : threads) {
    thread.join();
  }

  std::this_thread::sleep_for(
      std::chrono::milliseconds(100));  // Wait for async logging

  std::stringstream ss;
  std::ifstream file(log_file);
  ss << file.rdbuf();
  std::string log_output = ss.str();

  for (int i = 0; i < num_threads; ++i) {
    std::string debug_message =
        "Debug message from thread " + std::to_string(i);
    std::string info_message = "Info message from thread " + std::to_string(i);
    std::string error_message =
        "Error message from thread " + std::to_string(i);

    EXPECT_TRUE(log_output.find(debug_message) != std::string::npos)
        << "Thread " << i;
    EXPECT_TRUE(log_output.find(info_message) != std::string::npos)
        << "Thread " << i;
    EXPECT_TRUE(log_output.find(error_message) != std::string::npos)
        << "Thread " << i;
  }

  std::remove(log_file.c_str());
}
