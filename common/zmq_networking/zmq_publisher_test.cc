#include "zmq_publisher.h"

#include <gtest/gtest.h>
#include <string>

TEST(ZmqPublisherTest, SocketConnection) {
    std::string endpoint = "tcp://localhost:5555";
    ZmqPublisher pub(endpoint);
    EXPECT_TRUE(true);
}
