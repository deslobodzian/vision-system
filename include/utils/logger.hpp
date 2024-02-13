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

inline std::string log_seperator(const std::string& text) {
    const std::string line = "---------------------------------------------";
    const size_t line_width = line.length();

    size_t total_padding = line_width > text.length() ? line_width - text.length() : 0;

    size_t left_padding = total_padding / 2;
    size_t right_padding = total_padding - left_padding;

    return std::string(left_padding, '-') + text + std::string(right_padding, '-');
}

constexpr std::string_view extract_class_name(const char* pretty_function) {
    std::string_view pf = pretty_function;

    size_t params_start = pf.find('(');
    if (params_start == std::string_view::npos)
        return {}; // Not a valid function

    size_t colons = pf.rfind("::", params_start);
    if (colons == std::string_view::npos) {
        size_t name_start = pf.rfind(' ', params_start);
        name_start = (name_start == std::string_view::npos) ? 0 : name_start + 1;
        return pf.substr(name_start, params_start - name_start);
    }

    // Extract the class name from the substring before ::
    size_t class_name_end = colons;
    size_t class_name_start = pf.rfind(' ', colons);
    class_name_start = (class_name_start == std::string_view::npos) ? 0 : class_name_start + 1;

    return pf.substr(class_name_start, class_name_end - class_name_start);
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
