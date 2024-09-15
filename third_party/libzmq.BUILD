load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:private"],
)

cmake(
    name = "libzmq",
    cache_entries = {
        "CMAKE_INSTALL_PREFIX": "$$INSTALLDIR$$",
        "BUILD_SHARED_LIBS": "OFF",
        "BUILD_STATIC_LIBS": "ON",
        "WITH_PERF_TOOL": "OFF",
        "ZMQ_BUILD_TESTS": "OFF",
    },
    lib_source = ":all_srcs",
    out_static_libs = ["libzmq.a"],
    out_headers_only = True,
    includes = ["include"],
    generate_args = ["-GNinja"],
)

cc_library(
    name = "libzmq_cc",
    deps = [":libzmq"],
    includes = ["include"],
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
)
