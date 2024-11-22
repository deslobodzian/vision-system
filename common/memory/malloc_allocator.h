#ifndef VISION_SYSTEM_COMMON_MALLOC_ALLOCATOR_H
#define VISION_SYSTEM_COMMON_MALLOC_ALLOCATOR_H

#include "allocator.h"
#include <logger.h>
#include <cstdlib>
#include <cstring>

class MallocAllocator : public Allocator {
public:
 MallocAllocator() = default;
 MallocAllocator(const MallocAllocator&) = default;
 MallocAllocator(MallocAllocator&&) = delete;
 MallocAllocator& operator=(const MallocAllocator&) = default;
 MallocAllocator& operator=(MallocAllocator&&) = delete;
 ~MallocAllocator() override = default;

 void* alloc(size_t size) override {
   void* data = ::operator new(size);
   return data;
    }

    void free(void* ptr) override {
        LOG_DEBUG("Freeing ptr: ", ptr);
        ::operator delete(ptr);
    }

    void copy_in(void* dst, void* src, size_t size) override {
       std::memcpy(dst, src, size);
    }

    void copy_out(void* src, void* dst, size_t size) override {
       std::memcpy(dst, src,  size);
    }
};

#endif /* VISION_SYSTEM_COMMON_MALLOC_ALLOCATOR_H */
