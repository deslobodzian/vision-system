//
// Created by DSlobodzian on 4/21/2022.
//
#include "vision/tracked_target_info.hpp"

TrackedTargetInfo::TrackedTargetInfo() {}

TrackedTargetInfo::TrackedTargetInfo(float x, float y, float z, int id) {
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

TrackedTargetInfo::TrackedTargetInfo(const sl::ObjectData& object_data) {
    id_ = object_data.raw_label;
    x_ = object_data.position.x;
    y_ = object_data.position.y;
    z_ = object_data.position.z;
    vx_ = object_data.velocity.x;
    vy_ = object_data.velocity.y;
    vz_ = object_data.velocity.z;
}

float TrackedTargetInfo::get_x() const {
    return x_;
}

float TrackedTargetInfo::get_y() const {
    return y_;
}

float TrackedTargetInfo::get_z() const {
    return z_;
}

int TrackedTargetInfo::get_id() const {
    return id_;
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

std::vector<float> TrackedTargetInfo::get_vec() const {
    return {static_cast<float>(id_), x_, y_, z_, vx_, vy_, vz_};
}

TrackedTargetInfo::~TrackedTargetInfo() {

}





