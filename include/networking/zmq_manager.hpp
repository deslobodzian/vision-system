//
// Created by robot on 1/17/23.
//

#ifndef VISION_SYSTEM_ZMQ_MANAGER_HPP
#define VISION_SYSTEM_ZMQ_MANAGER_HPP

#include <string>
#include <map>
#include "zmq_publisher.hpp"

class ZmqManager {
public:
    ZmqManager();
    ~ZmqManager() {
        for (auto& [topic, publisher] : publishers_) {
            delete publisher;
        }
    }
    void create_publisher(const std::string& topic, const std::string& endpoint) {
        publishers_[topic] = new ZmqPublisher(endpoint);
    }

private:
    std::map<std::string, ZmqPublisher*> publishers_;
};

#endif //VISION_SYSTEM_ZMQ_MANAGER_HPP
