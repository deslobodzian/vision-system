#ifndef VISION_SYSTEM_COMMON_ALLOCATOR_H
#define VISION_SYSTEM_COMMON_ALLOCATOR_H
/*
 * Virutal class for all allocator types
 */
#include <cstddef>

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void* alloc(size_t size) = 0;
  virtual void free(void* ptr) = 0;
  virtual void copy_in(void* dst, void* src, size_t size) = 0;
  virtual void copy_out(void* src, void* dst, size_t size) = 0;
};

#endif /* VISION_SYSTEM_COMMON_ALLOCATOR_H */
