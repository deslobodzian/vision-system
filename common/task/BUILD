package(default_visibility = ["//visibility:public"])

cc_library(
    name = "task",
    srcs = ["task.cc"],
    hdrs = ["task.h"],
    visibility = [ "//visibility:public" ],
    deps = [
        "//common/logger:logger",
    ],
)

cc_test(
    name = "task_test",
    srcs = ["task_test.cc"],
    deps = [
        ":task",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
