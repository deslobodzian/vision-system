#include "malloc_allocator.h"

#include <logger.h>
#include <gtest/gtest.h>

TEST(MallocAllocatorTest, AllocatorTests) {
  MallocAllocator alloc{};

  constexpr size_t elements = 5;
  constexpr size_t size = elements * sizeof(int);
  int* data = nullptr;
  std::array<int, elements> dummy{0, 1, 2, 3, 4};

  data = static_cast<int*>(alloc.alloc(size));
  alloc.copy_out(dummy.data(), data, size);

  EXPECT_TRUE(std::equal(data, data + elements, dummy.data()));
}
