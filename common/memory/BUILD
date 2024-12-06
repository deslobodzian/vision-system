load("@rules_cuda//cuda:defs.bzl", "cuda_library", "requires_cuda")

package(default_visibility= ["//visibility:public"])

cc_library(
    name = "allocator",
    hdrs = ["allocator.h", "malloc_allocator.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//common/logger:logger"
    ]
)

cc_library(
    name = "device",
    hdrs = [
        "device.h", 
        "cpu_device.h",
        "device_manager.h",
    ],
    includes = ["."],
    defines = [] + select({
        "@rules_cuda//cuda:is_enabled": ["CUDA_ENABLED"],
        "//conditions:default": ["CUDA_DISABLED"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":allocator",
        "//common/logger:logger"
    ] + select({
        "@rules_cuda//cuda:is_enabled": ["@local_cuda//:cuda_runtime"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "alloc_tests",
    srcs = ["allocator_tests.cc"],
    deps = [
        ":allocator",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "device_tests",
    srcs = ["device_tests.cc"],
    deps = [
        ":device",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

