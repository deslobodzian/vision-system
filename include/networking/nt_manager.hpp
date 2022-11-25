//
// Created by deslobodzian on 11/23/22.
//

#ifndef VISION_SYSTEM_NT_MANAGER_HPP
#define VISION_SYSTEM_NT_MANAGER_HPP

#include "networktables/NetworkTableInstance.h"
#include "nt_publisher.hpp"
#include "nt_subscriber.hpp"
#include <map>

class NTManager {
public:
    NTManager() {
        inst_ = nt::NetworkTableInstance::GetDefault();
    }

    void add_publisher(publishable *p) {
        publisher_map_.emplace(std::pair<std::string, NTPublisher>(p->get_topic(), NTPublisher(inst_, "table", p)));
    }

    void add_subscriber(subscribable *s) {
        subscriber_map_.emplace(std::pair<std::string, NTSubscriber>(s->get_topic(), NTSubscriber(inst_, "table", s)));
    }

    void publish(subscribable *s) {
        publisher_map_.find(s->get_topic())->second.publish();
    }
private:
    nt::NetworkTableInstance inst_;
    std::unordered_map<std::string, NTPublisher> publisher_map_;
    std::unordered_map<std::string, NTSubscriber> subscriber_map_;
};

#endif //VISION_SYSTEM_NT_MANAGER_HPP
