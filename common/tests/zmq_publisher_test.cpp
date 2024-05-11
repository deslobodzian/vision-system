#include "zmq_publisher.hpp"
#include "logger.hpp"

#include <gtest/gtest.h>
#include <zmq.h>
#include <thread>
#include <chrono>
#include <string>

void simple_subscriber(const std::string& endpoint, std::string& out_message) {
    void* context = zmq_ctx_new();
    void* subscriber = zmq_socket(context, ZMQ_SUB);
    zmq_connect(subscriber, endpoint.c_str());
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);  // Subscribe to all messages

    char buffer[256];
    zmq_recv(subscriber, buffer, 255, 0);
    out_message = std::string(buffer);

    zmq_close(subscriber);
    zmq_ctx_destroy(context);
}
TEST(ZmqPublisherTest, PublishesStringMessagesCorrectly) {
    std::string endpoint = "tcp://*:5555";
    std::string expected_message = "Hello, World!";
    std::string received_message;

    // Start the subscriber in a separate thread
    std::thread sub_thread(simple_subscriber, "tcp://localhost:5555", std::ref(received_message));

    // Allow some time for the subscriber to set up
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create and use the publisher
    ZmqPublisher publisher(endpoint);
    publisher.publish_string("Hello, World!", expected_message);

    // Wait for the subscriber to receive the message
    sub_thread.join();

    EXPECT_EQ(received_message, expected_message);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
