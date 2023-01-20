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
    ZmqManager() = default;

    ~ZmqManager() {
        for (auto& [topic, publisher] : publishers_) {
            delete publisher;
        }
    }
    void create_publisher(publishable* p, const std::string& endpoint) {
        publishers_[p->get_topic()] = new ZmqPublisher(endpoint, p);
    }
    // this sends all the publishers in the map
    void send_publishers() {
        for (auto& [topic, publisher] : publishers_) {
            publisher->send();
        }
    }
private:
    std::map<std::string, ZmqPublisher*> publishers_;
};

#endif //VISION_SYSTEM_ZMQ_MANAGER_HPP
