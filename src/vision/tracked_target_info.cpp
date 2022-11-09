//
// Created by DSlobodzian on 4/21/2022.
//
#include "vision/tracked_target_info.hpp"

TrackedTargetInfo::TrackedTargetInfo() {}


double TrackedTargetInfo::get_x() const {
    return x_;
}

double TrackedTargetInfo::get_y() const {
    return y_;
}

double TrackedTargetInfo::get_z() const {
    return z_;
}

double TrackedTargetInfo::get_distance(double x_offset, double y_offset, double z_offset) {
    double x = pow(get_x() - x_offset, 2);
    double y = pow(get_y() - y_offset, 2);
    double z = pow(get_z() - z_offset, 2);
    return sqrt(x + y + z);
}

double TrackedTargetInfo::get_distance(const sl::Transform& offset) {
    return get_distance(offset.tx, offset.ty, offset.tz);
}

double TrackedTargetInfo::get_yaw_angle() {
    return atan((sqrt((get_x() * get_x()) + (get_y() + get_y())) / get_z()));
}

double TrackedTargetInfo::get_pitch_angle() {
    return atan((get_y() / get_z()));
}

std::string TrackedTargetInfo::to_packet() {
    return target_identity_ + ";" +
            std::to_string(x_) + ";" +
            std::to_string(y_) + ";" +
            std::to_string(z_) + ";";
}

TrackedTargetInfo::~TrackedTargetInfo() {

}




