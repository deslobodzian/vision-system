#ifndef VISION_SYSTEM_ZMQ_PUBLISHER_HPP
#define VISION_SYSTEM_ZMQ_PUBLISHER_HPP

#include <flatbuffers/flatbuffers.h>
#include <zmq.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <logger.h>

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& endpoint)
    : context_(zmq_ctx_new()), publisher_(zmq_socket(context_, ZMQ_PUB)) {
        if (!context_ || !publisher_) {
            throw std::runtime_error("Failed to create ZeroMQ context or socket.");
        }

        int rc = zmq_bind(publisher_, endpoint.c_str());
        if (rc != 0) {
            zmq_close(publisher_);
            zmq_ctx_destroy(context_);
            throw std::runtime_error("Failed to bind to endpoint: " + endpoint +
                                     ", Error: " + zmq_strerror(errno));
        }

        int sndhwm = 1000;
        zmq_setsockopt(publisher_, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
        LOG_DEBUG("Created publisher on: ", endpoint);
    }

    void send_message(const std::string& topic, const uint8_t* data,
                      size_t size) {
        zmq_msg_t topic_msg;
        zmq_msg_t data_msg;

        zmq_msg_init_size(&topic_msg, topic.size());
        memcpy(zmq_msg_data(&topic_msg), topic.data(), topic.size());

        zmq_msg_init_size(&data_msg, size);
        memcpy(zmq_msg_data(&data_msg), data, size);

        zmq_msg_send(&topic_msg, publisher_, ZMQ_SNDMORE);
        zmq_msg_send(&data_msg, publisher_, 0);

        zmq_msg_close(&topic_msg);
        zmq_msg_close(&data_msg);
    }

    ~ZmqPublisher() {
        zmq_close(publisher_);
        zmq_ctx_destroy(context_);
    }

private:
    void* context_;
    void* publisher_;
    std::mutex mtx_;

};

#endif /* VISION_SYSTEM_ZMQ_PUBLISHER_HPP */

