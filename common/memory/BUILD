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

cc_test(
    name = "alloc_tests",
    srcs = ["allocator_tests.cc"],
    deps = [
        ":allocator",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

