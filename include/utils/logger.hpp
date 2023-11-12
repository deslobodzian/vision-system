#ifndef VISION_SYSTEM_LOGGER_HPP
#define VISION_SYSTEM_LOGGER_HPP

#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace logger {

enum class LogLevel {
    ERROR,
    INFO,
    DEBUG
};

#if defined(LOG_LEVEL)
    #if LOG_LEVEL == 2
        constexpr LogLevel CURRENT_LOG_LEVEL = LogLevel::DEBUG;
    #elif LOG_LEVEL == 1
        constexpr LogLevel CURRENT_LOG_LEVEL = LogLevel::INFO;
    #elif LOG_LEVEL == 0
        constexpr LogLevel CURRENT_LOG_LEVEL = LogLevel::ERROR;
    #endif
#else
    constexpr LogLevel CURRENT_LOG_LEVEL = LogLevel::INFO;
#endif

constexpr std::string_view extract_class_name(const char* pretty_function) {
    std::string_view pf = pretty_function;
    size_t colons = pf.find("::");
    if (colons == std::string_view::npos)
        return {}; // Not a class member function

    // Reverse-find the beginning of the class name (either space or start of string)
    size_t begin = pf.rfind(' ', colons);
    begin = (begin == std::string_view::npos) ? 0 : begin + 1;

    // Trim everything before the class name
    pf.remove_prefix(begin);

    // Find the end of the class name (either colons or template parameters)
    size_t end = pf.find("::");
    if (end != std::string_view::npos) {
        // Check if we're dealing with template parameters '<'
        size_t template_params = pf.find('<', begin);
        if (template_params != std::string_view::npos && template_params < end) {
            end = template_params;
        }
        pf.remove_suffix(pf.size() - end);
    }
    return pf;
}


class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }

    template<typename... Args>
    void log(LogLevel level, const char* pretty_function, Args&&... args) {
        std::string_view class_name = extract_class_name(pretty_function);
        std::scoped_lock<std::mutex> lock(mutex_);
        std::ostringstream stream;
        switch (level) {
            case LogLevel::INFO:
                stream << "\033[1;37m[INFO]\033[0m"; break;
            case LogLevel::DEBUG:
                stream << "\033[1;33m[DEBUG]\033[0m"; break;
            case LogLevel::ERROR:
                stream << "\033[1;31m[ERROR]\033[0m"; break;
        }
        // Continue with class name and message in default formatting
        stream << "[" << class_name << "]: ";
        (stream << ... << std::forward<Args>(args));
        std::cout << stream.str() << std::endl;
    }

private:
    std::mutex mutex_;
};

#define LOG(level, ...) do { \
    if (level <= logger::CURRENT_LOG_LEVEL) { \
        logger::Logger::instance().log(level, __PRETTY_FUNCTION__, __VA_ARGS__); \
    } \
} while (0)

#define LOG_INFO(...) LOG(logger::LogLevel::INFO, __VA_ARGS__)
#define LOG_DEBUG(...) LOG(logger::LogLevel::DEBUG, __VA_ARGS__)
#define LOG_ERROR(...) LOG(logger::LogLevel::ERROR, __VA_ARGS__)


} // namespace logger

#endif // VISION_SYSTEM_LOGGER_HPP