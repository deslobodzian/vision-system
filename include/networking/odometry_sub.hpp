//
// Created by ubuntuvm on 11/23/22.
//

#ifndef VISION_SYSTEM_ODOMETRY_SUB_HPP
#define VISION_SYSTEM_ODOMETRY_SUB_HPP

#include "nt_subscriber.hpp"

struct odometry_subscribable : public subscribable {
public:
    double dx;
    double dy;
    double d_theta;
    void copy_vec(const std::vector<double> &data) {
        dx = data.at(0);
        dy = data.at(1);
        d_theta = data.at(2);
    }
    std::string get_topic() const{
        return topic_;
    }
    const std::string topic_ = "odometry";
};

#endif //VISION_SYSTEM_ODOMETRY_SUB_HPP
