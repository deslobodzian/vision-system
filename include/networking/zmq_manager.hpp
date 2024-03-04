#ifndef VISION_SYSTEM_ZMQ_MANAGER_HPP
#define VISION_SYSTEM_ZMQ_MANAGER_HPP


#include <map>
#include <string>
#include "zmq_publisher.hpp"
#include "zmq_subscriber.hpp"

class ZmqManager {
public:
    ZmqManager() = default;

    ~ZmqManager() {
    }

    void create_publisher(const std::string& name, const std::string& endpoint) {
        publishers_[name] = std::make_unique<ZmqPublisher>(endpoint);
    }

    void create_subscriber(const std::string& topic, const std::string& endpoint) {
        subscribers_[topic] = std::make_unique<ZmqSubscriber>(topic, endpoint);
    }

    ZmqPublisher& get_publisher(const std::string& name) {
        auto it = publishers_.find(name);
        if (it != publishers_.end()) {
            return *(it->second);
        } else {
            throw::std::runtime_error("Publisher not found: " + name);
        }
    }

    ZmqSubscriber& get_subscriber(const std::string& topic) {
        auto it = subscribers_.find(topic);
        if (it != subscribers_.end()) {
            return *(it->second);
        } else {
            throw::std::runtime_error("Subscriber not found: " + topic);
        }
    }

private:
    std::string publisher_endpoint_;
    std::map<std::string, std::unique_ptr<ZmqPublisher>> publishers_;
    std::map<std::string, std::unique_ptr<ZmqSubscriber>> subscribers_;

};
#endif /* VISION_SYSTEM_ZMQ_MANAGER_HPP */
