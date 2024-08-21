load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "libzmq",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "OFF",
        "ZMQ_BUILD_TESTS": "OFF",
    },
    lib_source = ":all_srcs",
    out_static_libs = ["libzmq.a"],
)
