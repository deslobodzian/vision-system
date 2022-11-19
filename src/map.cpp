//
// Created by DSlobodzian on 2/7/2022.
//

#include "map.hpp"

Map::Map() {
    Landmark l1(16.459252 - 13.2334, 8.22960 - 0.3683, 0);
    Landmark l2(16.459252 - 13.2334, 8.22960 - 2.85, 1);
    Landmark l3(16.45925 / 2.0, 8.22960 / 2.0, 2);
    Landmark l4(13.2334, 0.3683, 3);
    Landmark l5(13.2334, 2.85, 4);
    landmarks_.emplace_back(l1);
    landmarks_.emplace_back(l2);
    landmarks_.emplace_back(l3);
    landmarks_.emplace_back(l4);
    landmarks_.emplace_back(l5);
}

std::vector<Landmark> Map::get_landmarks() {
    return landmarks_;
}

