#ifndef VISION_SYSTEM_COMMON_BUFFER_H
#define VISION_SYSTEM_COMMON_BUFFER_H

template <typename T>
class Buffer {
    Buffer() = default;
    T* data() { return data_; }

private:
    T* data_;
};
#endif /* VISION_SYSTEM_COMMON_BUFFER_H */
