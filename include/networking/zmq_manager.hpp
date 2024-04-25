#ifndef VISION_SYSTEM_ZMQ_MANAGER_HPP
#define VISION_SYSTEM_ZMQ_MANAGER_HPP

#include "zmq_publisher.hpp"
#include "zmq_subscriber.hpp"
#include <map>
#include <string>

class ZmqManager {
 public:
  ZmqManager() = default;

  ~ZmqManager() {}

  void create_publisher(const std::string& name, const std::string& endpoint) {
    publishers_[name] = std::make_unique<ZmqPublisher>(endpoint);
  }

  void create_subscriber(const std::string& topic,
                         const std::string& endpoint) {
    subscribers_[topic] = std::make_unique<ZmqSubscriber>(topic, endpoint);
  }

  ZmqPublisher& get_publisher(const std::string& name) {
    return *publishers_.at(name);
  }

  ZmqSubscriber& get_subscriber(const std::string& topic) {
    return *subscribers_.at(topic);
  }

 private:
  std::map<std::string, std::unique_ptr<ZmqPublisher>> publishers_;
  std::map<std::string, std::unique_ptr<ZmqSubscriber>> subscribers_;
};
#endif /* VISION_SYSTEM_ZMQ_MANAGER_HPP */
