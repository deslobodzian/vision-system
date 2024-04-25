#include "networking/zmq_manager.hpp"
#include "vision_pose_array_generated.h"
#include <atomic>
#include <iostream>
#include <memory>
#include <networktables/NetworkTable.h>
#include <networktables/NetworkTableEntry.h>
#include <networktables/NetworkTableInstance.h>
#include <string>
#include <thread>

void zmq_objects_subscriber_thread(
    ZmqManager& zmqManager, const std::string& topic,
    std::shared_ptr<nt::NetworkTable> vision_table,
    std::atomic<bool>& running) {
  using namespace std::chrono_literals;
  ZmqSubscriber& subscriber = zmqManager.get_subscriber(topic);
  constexpr int VISION_TIMEOUT_MS = 500;

  auto last_message_time = std::chrono::steady_clock::now();

  while (running.load()) {
    try {
      auto message = subscriber.receive();
      if (message) {
        const auto& [topic, data] = *message;
        LOG_DEBUG("Topic got: ", topic);
        const Messages::VisionPoseArray* vision_pose_array =
            Messages::GetVisionPoseArray(data.data());
        int numPoses = vision_pose_array->poses()->size();
        last_message_time = std::chrono::steady_clock::now();
        vision_table->GetEntry("num_objects").SetDouble(numPoses);
        vision_table->GetEntry("vision_poses_raw")
            .SetRaw(std::span(
                reinterpret_cast<const uint8_t*>(
                    std::as_bytes(
                        std::span(static_cast<const char*>(data.data()),
                                  data.size()))
                        .data()),
                data.size()));
      } else {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_message_time)
                .count();
        if (elapsed_time > VISION_TIMEOUT_MS) {
          vision_table->GetEntry("num_objects").SetDouble(0);
          vision_table->GetEntry("vision_poses_raw")
              .SetRaw(std::vector<uint8_t>());
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Error in zmqObjectsSubscriberThread: " << e.what()
                << std::endl;
    }
  }
}

int main() {
  try {
    nt::NetworkTableInstance inst = nt::NetworkTableInstance::GetDefault();
    constexpr int TEAM = 5687;
    inst.StartClient4("VisionProcessor");
    inst.SetServerTeam(TEAM);
    auto vision_table = inst.GetTable("VisionProcessor");

    auto zmq_manager = std::make_unique<ZmqManager>();
    zmq_manager->create_subscriber("Objects", "ipc:///home/orin/tmp/vision/0");

    std::atomic<bool> running{true};

    {
      std::thread subscriber_thread(zmq_objects_subscriber_thread,
                                    std::ref(*zmq_manager), "Objects",
                                    vision_table, std::ref(running));
      subscriber_thread.join();
    }

    zmq_manager.reset();
    inst.StopClient();
  } catch (const std::exception& e) {
    std::cerr << "Error in main: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
