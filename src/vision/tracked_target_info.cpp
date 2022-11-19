//
// Created by DSlobodzian on 4/21/2022.
//
#include "vision/tracked_target_info.hpp"

TrackedTargetInfo::TrackedTargetInfo() {}

TrackedTargetInfo::TrackedTargetInfo(double x, double y, double z, int id) {
    id_ = id;
    x_ = x;
    y_ = y;
    z_ = z;
}

TrackedTargetInfo::TrackedTargetInfo(sl::Pose &pose, int id) {
    id_ = id;
    x_ = pose.getTranslation().tx;
    y_ = pose.getTranslation().ty;
    z_ = pose.getTranslation().tz;
}

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
double TrackedTargetInfo::get_distance() {
    return get_distance(0, 0, 0);
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
    return std::to_string(id_) + ";" +
            std::to_string(x_) + ";" +
            std::to_string(y_) + ";" +
            std::to_string(z_) + ";";
}

std::span<double> TrackedTargetInfo::target_info_vector() {
    std::vector<double> tmp{x_, y_, z_, get_yaw_angle(), id_};
    return std::span<double> {tmp};
}

TrackedTargetInfo::~TrackedTargetInfo() {

}





