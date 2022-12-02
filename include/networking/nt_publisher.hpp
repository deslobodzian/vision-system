//
// Created by ubuntuvm on 11/16/22.
//

#ifndef VISION_SYSTEM_NT_PUBLISHER_HPP
#define VISION_SYSTEM_NT_PUBLISHER_HPP

#include <mutex>
#include <networktables/RawTopic.h>
#include <span>
#include "vision/tracked_target_info.hpp"

struct publishable {
    virtual std::span<uint8_t> to_span() {return std::span<uint8_t>{};};
    virtual std::string get_topic() const = 0;
    virtual void encode(uint8_t* buffer) = 0;
    std::mutex publishable_mtx_;
};
class NTPublisher {
public:
    NTPublisher() = default;
    NTPublisher(const nt::NetworkTableInstance &instance, const std::string &table, publishable *p) {
        auto nt_table = instance.GetTable(table);
        p_ = p;
        pub_ = nt_table->GetRawTopic(p->get_topic()).Publish("");
    }

    void publish() {
        std::lock_guard<std::mutex> lock(p_->publishable_mtx_);
        pub_.Set(p_->to_span());
    }

private:
    nt::RawPublisher pub_;
    publishable* p_;
};

#endif //VISION_SYSTEM_NT_PUBLISHER_HPP
