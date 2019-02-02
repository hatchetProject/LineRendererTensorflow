genrule(
    name = "line_renderer_kernel",
    srcs = ["line_renderer_grad.cc",
            "rasterize_lines_op.cc",
            "rasterize_lines_impl.cc",
            "rasterize_lines_impl.h"],
    outs = ["rasterize_triangles_kernel.so"],
    cmd = "TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())');\
           TF_LIB=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'); \
           g++ -std=c++11 -shared $(SRCS) -o $@ -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$$TF_INC -I$$TF_INC/external/nsync/public -L$$TF_LIB -ltensorflow_framework -O2",
    visibility = ["//tf_line_renderer:__subpackages__"],
)


