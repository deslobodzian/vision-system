#include <iostream>
#include <string>
#include <thread>
#include <networktables/NetworkTable.h>
#include <networktables/NetworkTableEntry.h>
#include <networktables/NetworkTableInstance.h>
#include "networking/zmq_manager.hpp"
#include "vision_pose_array_generated.h"

void zmq_objects_subscriber_thread(ZmqManager& zmqManager, const std::string& topic, std::shared_ptr<nt::NetworkTable> vision_table, bool& running) {
    ZmqSubscriber& subscriber = zmqManager.get_subscriber(topic);

    while (running) {
        try {
            auto message = subscriber.receive();
            if (message) {
                const auto& [topic, data] = *message;
                LOG_DEBUG("Topic got: ", topic);
                const Messages::VisionPoseArray* vision_pose_array = Messages::GetVisionPoseArray(data.data());
                int numPoses = vision_pose_array->poses()->size();
                vision_table->GetEntry("num_objects").SetDouble(numPoses);
                // This is cursed
                vision_table->GetEntry("vision_poses_raw").SetRaw(
                    std::span(reinterpret_cast<const uint8_t*>(std::as_bytes(std::span(static_cast<const char*>(data.data()), data.size())).data()), data.size())
                );
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error in zmqObjectsSubscriberThread: " << e.what() << std::endl;
        }
    }
}

int main() {
    nt::NetworkTableInstance inst = nt::NetworkTableInstance::GetDefault();
    
    constexpr int TEAM = 5687;
    inst.StartClient4("VisionProcessor");
    inst.SetServerTeam(TEAM);

    auto vision_table = inst.GetTable("VisionProcessor");

    ZmqManager zmq_manager;
    zmq_manager.create_subscriber("Objects", "ipc:///home/orin/tmp/vision/0");

    bool running = true;
    std::thread subscriber_thread(zmq_objects_subscriber_thread, std::ref(zmq_manager), "Objects", vision_table, std::ref(running));

    subscriber_thread.join();

    inst.StopClient();

    return 0;
}

