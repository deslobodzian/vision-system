#include "zmq_publisher.h"
#include "zmq_subscriber.h"

#include <gtest/gtest.h>
#include <string>
#include <thread>


void publisher_send_message(const std::string& topic, ZmqPublisher* pub, const std::vector<uint8_t>& msg) {
    //Make sure to wait for sub to latch
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    for (int i = 0; i < 5; i++) {
        pub->send_message(topic, msg.data(), msg.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void subscriber_receive(ZmqSubscriber* sub, const std::vector<uint8_t>& sent_data) {
    std::vector<bool> correct_received;

    for (int i = 0; i < 5; i++) {
        auto rec = sub->receive_message();
        LOG_DEBUG("Message received is: ", rec.has_value());
        if (rec.has_value()) {
            LOG_DEBUG("Message received topic: ", rec->first);
            auto rec_data = rec->second;
            LOG_DEBUG("Data length: ", rec_data.size());
            bool same_data = rec_data == sent_data;
            EXPECT_TRUE(same_data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


TEST(ZmqTests, PubSubMessaging) {
    std::string endpoint = "tcp://localhost:5556";

    ZmqPublisher pub(endpoint);
    ZmqSubscriber sub(endpoint, {"test"});

    std::vector<std::thread> threads;
    std::vector<uint8_t> test_message = {0x48, 0x65, 0x6C, 0x6C, 0x6F}; // "Hello" in hex

    threads.emplace_back(publisher_send_message, "test", &pub, test_message);
    threads.emplace_back(subscriber_receive, &sub, test_message);

    for (auto &thread : threads) {
        thread.join();
    }
    //FAIL() << "Failed to receive the message after multiple attempts";
}
