package(default_visibility = ["//visibility:public"])

cc_library(
    name = "logger",
    hdrs = ["logger.h"],
    includes = ["."],
    visibility = [ "//visibility:public" ],
)


cc_test(
    name = "logger_test",
    srcs = ["logger_test.cc"],
    deps = [
        ":logger",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

