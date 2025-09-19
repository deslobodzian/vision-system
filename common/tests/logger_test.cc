#include "logger.h"


#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>

TEST_CASE("LoggerTest - Levels", "[logger][level]") {
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
  REQUIRE(log_output.find("Debug message") != std::string::npos);
  REQUIRE(log_output.find("Info message") != std::string::npos);
  REQUIRE(log_output.find("Error message") != std::string::npos);
}

TEST_CASE("LoggerTest - LogFile", "[logger][file]") {
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

  REQUIRE(log_output.find("Debug message") != std::string::npos);
  REQUIRE(log_output.find("Info message") != std::string::npos);
  REQUIRE(log_output.find("Error message") != std::string::npos);

  std::remove(log_file.c_str());
}

void log_from_thread(int thread_id) {
  LOG_DEBUG("Debug message from thread ", thread_id);
  LOG_INFO("Info message from thread ", thread_id);
  LOG_ERROR("Error message from thread ", thread_id);
}

TEST_CASE("LoggerTest - MultiThreadedLogging", "[logger][multithread]") {
    // Arrange
    logger::Logger::instance().set_log_level(logger::LogLevel::DEBUG);

    const std::string log_file = "test_log_multithread.txt";
    logger::Logger::instance().set_log_file(log_file);

    constexpr int num_threads = 5;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Act: spawn workers that log
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(log_from_thread, i);
    }
    for (auto& t : threads) t.join();

    // If your logger is async and exposes a flush, call it here:
    // logger::Logger::instance().flush();

    // Deterministic wait: poll file until all messages are present or we time out.
    auto all_expected_present = [&](const std::string& contents) {
        for (int i = 0; i < num_threads; ++i) {
            const std::string dbg = "Debug message from thread " + std::to_string(i);
            const std::string inf = "Info message from thread " + std::to_string(i);
            const std::string err = "Error message from thread " + std::to_string(i);
            if (contents.find(dbg) == std::string::npos) return false;
            if (contents.find(inf) == std::string::npos) return false;
            if (contents.find(err) == std::string::npos) return false;
        }
        return true;
    };

    std::string log_output;
    {
        using namespace std::chrono_literals;
        const auto deadline = std::chrono::steady_clock::now() + 2s; // max 2s
        do {
            std::ifstream file(log_file);
            REQUIRE(file.is_open()); // Fail fast if we couldn't open
            std::ostringstream ss;
            ss << file.rdbuf();
            log_output = std::move(ss).str();

            if (all_expected_present(log_output)) break;
            std::this_thread::sleep_for(50ms);
        } while (std::chrono::steady_clock::now() < deadline);
    }

    // Assert
    for (int i = 0; i < num_threads; ++i) {
        const std::string dbg = "Debug message from thread " + std::to_string(i);
        const std::string inf = "Info message from thread " + std::to_string(i);
        const std::string err = "Error message from thread " + std::to_string(i);

        CAPTURE(i); // Will print i if any REQUIRE fails
        REQUIRE(log_output.find(dbg) != std::string::npos);
        REQUIRE(log_output.find(inf) != std::string::npos);
        REQUIRE(log_output.find(err) != std::string::npos);
    }

    // Cleanup
    std::remove(log_file.c_str());
}
