//
// Created by ubuntuvm on 11/16/22.
//

#ifndef VISION_SYSTEM_NT_PUBLISHER_HPP
#define VISION_SYSTEM_NT_PUBLISHER_HPP

#include <networktables/DoubleArrayTopic.h>
#include "vision/tracked_target_info.hpp"

class NTPublisher {
public:
    NTPublisher();
    NTPublisher(nt::NetworkTableInstance &instance, std::string table) {
        auto nt_table = instance.GetTable(table);
        pub_ = nt_table->GetDoubleArrayTopic("target_data").Publish();
    }
    void publish_target_data(TrackedTargetInfo target_info) {
        Set(target_info.target_info_vector());
    }
private:
    nt::DoubleArrayPublisher pub_;

};

#endif //VISION_SYSTEM_NT_PUBLISHER_HPP
