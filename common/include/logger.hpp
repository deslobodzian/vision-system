#ifndef VISION_SYSTEM_COMMON_LOGGER_HPP
#define VISION_SYSTEM_COMMON_LOGGER_HPP

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace logger {

enum class LogLevel { ERROR, INFO, DEBUG };

constexpr std::string_view extract_class_name(const char *pretty_function) {
  std::string_view pf = pretty_function;

  size_t params_start = pf.find('(');
  if (params_start == std::string_view::npos) {
    return {};  // Not a valid function
  }

  size_t colons = pf.rfind("::", params_start);
  if (colons == std::string_view::npos) {
    size_t name_start = pf.rfind(' ', params_start);
    name_start = (name_start == std::string_view::npos) ? 0 : name_start + 1;
    return pf.substr(name_start, params_start - name_start);
  }

  // Extract the class name from the substring before ::
  size_t class_name_end = colons;
  size_t class_name_start = pf.rfind(' ', colons);
  class_name_start =
      (class_name_start == std::string_view::npos) ? 0 : class_name_start + 1;

  return pf.substr(class_name_start, class_name_end - class_name_start);
}

class Logger {
 public:
  static Logger &instance() {
    static Logger instance;
    return instance;
  }

  void set_log_file(const std::string &filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_.is_open()) {
      log_file_.close();
    }
    log_file_name_ = filename;
    log_file_.open(log_file_name_, std::ios::out | std::ios::app);
  }

  void set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_log_level_ = level;
  }

  void flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_.is_open()) {
      log_file_.flush();
    }
  }

  template <typename... Args>
  void log(LogLevel level, const char *pretty_function, Args &&...args) {
    if (level > current_log_level_) {
      return;
    }

    std::string_view class_name = extract_class_name(pretty_function);
    std::ostringstream stream;

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = std::localtime(&time_t);
    char date_time[20];
    std::strftime(date_time, sizeof(date_time), "%Y-%m-%d %H:%M:%S", tm);

    stream << date_time << " ";

    switch (level) {
      case LogLevel::INFO:
        stream << "\033[1;37m[INFO]\033[0m"; // Bold White
        break;
      case LogLevel::DEBUG:
        stream << "\033[1;33m[DEBUG]\033[0m"; // Bold Yellow
        break;
      case LogLevel::ERROR:
        stream << "\033[1;31m[ERROR]\033[0m"; // Bold Red
        break;
    }

    stream << "[" << class_name << "]: ";
    (stream << ... << std::forward<Args>(args));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::string log_message = stream.str();
      log_queue_.push_back(log_message);

      if (!log_file_name_.empty()) {
        file_log_queue_.push_back(log_message);
      }
    }

    cv_.notify_one();
  }

  ~Logger() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_logging_ = true;
    }
    cv_.notify_one();
    if (logging_thread_.joinable()) {
      logging_thread_.join();
    }
  }

 private:
  Logger() { logging_thread_ = std::thread(&Logger::log_thread, this); }

  void log_thread() {
    while (true) {
      std::vector<std::string> local_queue;
      std::vector<std::string> local_file_queue;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return !log_queue_.empty() || !file_log_queue_.empty() ||
                 stop_logging_;
        });
        if (stop_logging_ && log_queue_.empty() && file_log_queue_.empty()) {
          break;
        }
        local_queue.swap(log_queue_);
        local_file_queue.swap(file_log_queue_);
      }

      for (const auto &message : local_queue) {
        std::cout << message << std::endl;
      }

      if (!local_file_queue.empty()) {
        if (!log_file_.is_open()) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!log_file_.is_open()) {
            log_file_.open(log_file_name_, std::ios::out | std::ios::app);
          }
        }

        if (log_file_.is_open()) {
          for (const auto &message : local_file_queue) {
            log_file_ << message << std::endl;
          }
          log_file_.flush();
        }
      }
    }
  }

  std::mutex mutex_;
  std::vector<std::string> log_queue_;
  std::thread logging_thread_;
  std::condition_variable cv_;
  std::atomic<bool> stop_logging_{false};
  std::vector<std::string> file_log_queue_;
  std::string log_file_name_;
  std::ofstream log_file_;
  LogLevel current_log_level_ = LogLevel::DEBUG;
};

#define LOG(level, ...)                                                      \
  do {                                                                       \
    logger::Logger::instance().log(level, __PRETTY_FUNCTION__, __VA_ARGS__); \
  } while (0)

#define LOG_INFO(...) LOG(logger::LogLevel::INFO, __VA_ARGS__)
#define LOG_DEBUG(...) LOG(logger::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_ERROR(...) LOG(logger::LogLevel::ERROR, __VA_ARGS__)

}  // namespace logger

#endif  // VISION_SYSTEM_COMMON_LOGGER_HPP
