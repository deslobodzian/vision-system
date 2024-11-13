#ifndef VISION_SYSTEM_COMMON_ALLOCATOR_H
#define VISION_SYSTEM_COMMON_ALLOCATOR_H

/*
 * Virutal class for all allocator types
 */
class Allocator {
 public:
  virtual void* alloc(size_t size) = 0;
  virtual void free(void* ptr) = 0;
  virtual int copy_to(void* dst, void* src, size_t size) = 0;
  virtual int copy_from(void* dst, void* src, size_t size) = 0;
};

#endif /* VISION_SYSTEM_COMMON_ALLOCATOR_H */
