exports_files(
    [
        "LICENSE",
        "ACKNOWLEDGEMENTS",
    ],
)

cc_library(
    name = "test1",
    hdrs = ["out.h"],
    srcs = ["out_model.o", "out_helper.o"],
)

cc_binary(
    name = "ssd_incep_test",
    srcs = ["ssd_incep_preprocessor.cpp"],
    deps = [
	":test1",
	"//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
    ],
)
