//
// Created by DSlobodzian on 2/7/2022.
//
#pragma once

#include <iostream>
#include <vector>

#define FIELD_LENGTH

struct Landmark {
    double x;
    double y;
    int tag_id;
    Landmark(double x_pos, double y_pos, int id) {
        x = x_pos;
        y = y_pos;
        tag_id = id;
    }
};

class Map {

private:
    std::vector<Landmark> landmarks_;

public:
    Map();
    ~Map() = default;

    std::vector<Landmark> get_landmarks();

};


