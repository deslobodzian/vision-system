cc_library(
    name = "zmq_networking",
    hdrs = ["zmq_publisher.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@libzmq//:libzmq",
    ],
)

cc_test(
    name = "zmq_publisher_test",
    srcs = ["zmq_publisher_test.cc"],
    deps = [
        ":zmq_networking",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
