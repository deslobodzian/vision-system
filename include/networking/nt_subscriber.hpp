//
// Created by ubuntuvm on 11/23/22.
//

#ifndef VISION_SYSTEM_NT_SUBSCRIBER_HPP
#define VISION_SYSTEM_NT_SUBSCRIBER_HPP

#include <networktables/RawTopic.h>
#include <mutex>

struct subscribable {
    virtual void copy_vector(const std::vector<uint8_t> &data) = 0;
    virtual std::string get_topic() const=0;
    std::mutex subscriber_mtx_;
};

class NTSubscriber{
public:
    NTSubscriber() = default;
    NTSubscriber(const nt::NetworkTableInstance &instance, const std::string &table, subscribable *s) {
        auto nt_table = instance.GetTable(table);
        s_ = s;
        sub_ = nt_table->GetRawTopic(s_->get_topic()).Subscribe({}, {});
    }

    void get() {
        const std::lock_guard<std::mutex> lock(s_->subscriber_mtx_);
        s_->copy_vector(sub_.Get());
    }

private:
    nt::RawSubscriber sub_;
    subscribable* s_;
};
#endif //VISION_SYSTEM_NT_SUBSCRIBER_HPP
