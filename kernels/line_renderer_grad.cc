#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace tf_line_renderer{

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;

REGISTER_OP("LineRendererGrad")
    .Input("dline_dv_parameter: float32")
    .Input("pixel_count: int32")
    .Output("dline_dvertices: float32")
    .Doc(R"doc(
dline_dv_parameter: The output from rasterize_lines_op.cc.
pixel_count: Also the output form rasterize_lines_op.cc.
dline_dvertices: The wanted gradient for the two points on the line. A tensor with shape(4, ).
)doc");


class LineRendererGradOp : public OpKernel{
    public:
        explicit LineRendererGradOp(OpKernelConstruction* context) : OpKernel(context){}
        ~LineRendererGradOp() override {}
    
    void Compute(OpKernelContext* context) override {
        const Tensor& dline_dv_parameter_tensor = context->input(0);
        OP_REQUIRES(context, PartialTensorShape({-1, 6}).IsCompatibleWith(dline_dv_parameter_tensor.shape()), InvalidArgument("LineRendererGrad expects dline_dv_parameter shape to be (-1, 6), actually(200, 6)"));
       auto dline_dv_parameter_flat = dline_dv_parameter_tensor.flat<float>();
       const float* dline_dv_parameter = dline_dv_parameter_flat.data();


        const Tensor& pixel_count_tensor = context->input(1);
        OP_REQUIRES(context, PartialTensorShape({1, }).IsCompatibleWith(pixel_count_tensor.shape()), InvalidArgument("LineRendererGrad expects pixel_count to have shape (1, )"));
        auto pixel_count_flat = pixel_count_tensor.flat<int>();
        const int* pixel_count = pixel_count_flat.data();


        Tensor* df_dvertices_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({4, }), &df_dvertices_tensor));
        auto df_dvertices_flat = df_dvertices_tensor->flat<float>();
        float* dline_dvertices = df_dvertices_flat.data();
        
        const int pixel_num = pixel_count[0];
        std::fill(dline_dvertices, dline_dvertices + 4, 0.0f);

        for (int i = 0; i < pixel_num; ++i){
            const float dline_dv_param_1 = dline_dv_parameter[i*6 + 0];
            const float dline_dv_param_2 = dline_dv_parameter[i*6 + 1];
            const float dline_dvx_1 = dline_dv_parameter[i*6 + 2];
            const float dline_dvy_1 = dline_dv_parameter[i*6 + 3];
            const float dline_dvx_2 = dline_dv_parameter[i*6 + 4];
            const float dline_dvy_2 = dline_dv_parameter[i*6 + 5];
            
            const float df_dvertices1_x = dline_dv_param_1 * dline_dvx_1;
            const float df_dvertices1_y = dline_dv_param_1 * dline_dvy_1;
            const float df_dvertices2_x = dline_dv_param_2 * dline_dvx_2;
            const float df_dvertices2_y = dline_dv_param_2 * dline_dvy_2;
            dline_dvertices[0] += df_dvertices1_x;
            dline_dvertices[1] += df_dvertices1_y;
            dline_dvertices[2] += df_dvertices2_x;
            dline_dvertices[3] += df_dvertices2_y;
        
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("LineRendererGrad").Device(DEVICE_CPU), LineRendererGradOp);

}     //namespace tf_line_renderer
