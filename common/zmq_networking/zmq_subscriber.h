#ifndef VISION_SYSTEM_COMMON_ZMQ_SUBSCRIBER_H
#define VISION_SYSTEM_COMMON_ZMQ_SUBSCRIBER_H

#include <zmq.h>
#include <string>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <optional>

#include "logger.h"

class ZmqSubscriber {
public:
    ZmqSubscriber(const std::string& endpoint, std::initializer_list<std::string> topics) :
        ctx_(zmq_ctx_new()),
        subscriber_(zmq_socket(ctx_, ZMQ_SUB)),
        endpoint_(endpoint),
        topics_(topics) {
        if (!ctx_ || !subscriber_) {
            throw std::runtime_error("Failed to create ZeroMQ context or socket");
        }

        int rc = zmq_connect(subscriber_, endpoint.c_str());
        if (rc) {
            zmq_close(subscriber_);
            zmq_ctx_destroy(ctx_);
        }

        // I will handle topics
        rc = zmq_setsockopt(subscriber_, ZMQ_SUBSCRIBE, 0, 0);
        if (rc) {
            LOG_ERROR("Failed to subscriber to endpoint ", endpoint_);
        } else {
            LOG_DEBUG("Subscriber with endpoint {", endpoint_, "}");
        }
    }
    ~ZmqSubscriber() {
        LOG_DEBUG("Destroying subscriber: ", endpoint_);
        zmq_close(subscriber_);
        zmq_ctx_destroy(ctx_);
    }

    std::optional<std::pair<std::string, std::vector<uint8_t>>> receive_message() {
        int rc = 0;
        zmq_msg_t topic_msg, data_msg;
        zmq_msg_init(&topic_msg);
        zmq_msg_init(&data_msg);

        rc = zmq_msg_recv(&topic_msg, subscriber_, 0);
        LOG_DEBUG("Receive topic return code: ", rc); if (rc == -1) {
            std::string err  = "Failed to receive message to endpoint: " + endpoint_ +", Error: " + zmq_strerror(errno);
            LOG_ERROR(err);
            if (errno == EAGAIN) {
                // No message available
                zmq_msg_close(&topic_msg);
                zmq_msg_close(&data_msg);
                return std::nullopt;
            }
            throw std::runtime_error("Failed to receive topic message");
        }

        rc = zmq_msg_recv(&data_msg, subscriber_, 0);
        if (rc == -1) {
            throw std::runtime_error("Failed to receive data message");
        }

        std::string topic(static_cast<char*>(zmq_msg_data(&topic_msg)), zmq_msg_size(&topic_msg));
        std::vector<uint8_t> data(static_cast<uint8_t*>(zmq_msg_data(&data_msg)),
                                  static_cast<uint8_t*>(zmq_msg_data(&data_msg)) + zmq_msg_size(&data_msg));

        zmq_msg_close(&topic_msg);
        zmq_msg_close(&data_msg);

        return std::make_pair(topic, data);
    }

private:
    void* ctx_;
    void* subscriber_;
    std::string endpoint_;
    std::vector<std::string> topics_;

};
#endif /* VISION_SYSTEM_COMMON_ZMQ_SUBSCRIBER_H */

