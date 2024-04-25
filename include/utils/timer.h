#ifndef VISION_SYSTEM_TIMER_H
#define VISION_SYSTEM_TIMER_H

#include <chrono>

class Timer {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;
  using Duration = std::chrono::duration<float>;

  explicit Timer() : start_(Clock::now()) {}

  void start() { start_ = Clock::now(); }

  float get_ms() { return get_duration<std::chrono::milliseconds>().count(); }

  float get_nanoseconds() {
    return get_duration<std::chrono::nanoseconds>().count();
  }

  float get_seconds() { return get_duration<std::chrono::seconds>().count(); }

private:
  TimePoint start_;

  template <typename DurationType> DurationType get_duration() {
    return std::chrono::duration_cast<DurationType>(Clock::now() - start_);
  }
};
#endif /* VISION_SYSTEM_TIMER_H */