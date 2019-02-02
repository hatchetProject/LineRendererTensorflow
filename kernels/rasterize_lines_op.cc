#include <algorithm>
#include <vector>
#include <cmath>
#include <math.h>
#include <cmath>

//#include "rasterize_lines_impl.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


namespace tf_line_renderer{
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::int32;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;


REGISTER_OP("RasterizeLines")
        .Input("vertice_1: float32")
        .Input("vertice_2: float32")
        .Attr("image_width: int")
        .Attr("image_height: int")
        .Output("dline_dv_parameter: float32")
        .Output("pixel_count: int32")
        .Output("image_output: int32")
        .Doc(R"doc(
vertice_1: An 1-D tensor with shape (2, ), representing value (x_1, y_1). It is the starting point of the line.
vertice_2: An 1-D tensor with shape (2, ), representing value (x_2, y_2). It is the end ponit of the line.
image_width: positive int attribute specifying the width of the input image(/line);
image_height: positive int attrbute specifying the height of the input image(/line);
The 2 attributes limits the boundaries and points out of this boundary will be ignored.

pixel_count: The number of pixels on the line. With shape (1, )
dline_dv_parameter: A 2-D tensor with shape (200, 6). 200 is the imagined maximum value of pixels, as the graph has 48*48 = 2304 
    pixels in total. The values in each row are: Gradient of the pixel over vertice_1; Gradient of the pixel over vertice_2;
    The dline_dvx over vetice_1; The dline_dvy over vertice_1; The dline_dvx over vertice_2; The dline_dvy over vertice_2.
    The values in the tensor does not inlude the starting and ending point.
image_output: Including the coordinates of the pixels on the rasterized line, we take shape(200, 2) for this tensor.

)doc");


namespace {

inline int* DrawBresenhamline(int x0, int y0, int x1, int y1)
{
// Returns the pixel coordinates on the rendered line
// Including the starting point and the ending point

// There is a BUG here!!! Try with DrawBresenhamline(3, 4, 1, 2), it will output nothing!
// Remember to debug this error!!!
	int dx = x1 - x0;
	int dy = y1 - y0;
	int ux = dx >0 ?1:-1;
	int uy = dy >0 ?1:-1;
	int dx2 = dx <<1;
	int dy2 = dy <<1;
    int* res = new int[10];
    int count = 0;
	if(abs(dx)>abs(dy)){
		int e = -dx;
		int x = x0;
		int y = y0;
		for (x = x0; x < x1;x+=ux)
		{
			
            res[2*count] = x;
            res[2*count+1] = y;
            count++;
			e=e + dy2;
			if (e > 0)
			{
			 	y += uy;
				e= e - dx2;
			}
		}
	}
	else if (abs(dx) <= abs(dy) && abs(dx) != 0)
	{
		int e = -dy; 
		int x = x0;
		int y = y0;
		for (y = y0; y < y1;y += uy)
		{
			
            res[2*count] = x;
            res[2*count+1] = y;
            count++;
			e=e + dx2;
			if (e > 0)
			{
			 	x += ux;
				e= e - dy2;
			}
		}
	}
    else
    {
        uy = y0 < y1? 1 : -1;
        for (int y = y0; y < y1; y += uy){
            res[2*count] = x0;
            res[2*count+1] = y;
            count++;
        }
    }
    int result[2*count-2];
    for (int i = 0; i < 2*count-2; ++i)
        result[i] = res[i+2];
    delete []res;
    return result;
}




void RasterizeLinesImpl(const float* vertice_1, const float* vertice_2, 
                        float* dline_dv_parameter, int32* pixel_count, int32* image_output){
    // Input: vertice_1, vertice_2, both with shape (2, )
    // Output: dline_dv_parameter, pixel_count
    int vertice_1_x = vertice_1[0];
    int vertice_1_y = vertice_1[1];
    int vertice_2_x = vertice_2[0];
    int vertice_2_y = vertice_2[1];
    int* vertices = DrawBresenhamline(vertice_1_x, vertice_1_y, vertice_2_x, vertice_2_y);
    pixel_count[0] = (sizeof(vertices)/sizeof(vertices[0]))/2;
    for (int i = 0; i < pixel_count[0]; ++i){
        const int pixel_x = vertices[i*2];
        const int pixel_y = vertices[i*2+1];
        // The grad of pixel and vertice_1
        const float grad_1 = float(pixel_y - vertice_1_y)/(pixel_x - vertice_1_x);
        const float grad_2 = float(pixel_y - vertice_2_y)/(pixel_x - vertice_2_x);
        
        // Caculating dline_dvx,dline_dvy                
        // dline_dvx = x/sqrt(x^2 + y^2), dline_dvy = y/sqrt(x^2 + y^2)
        // x, y are the length of the line(pixel to starting/ending point)
        const float length_total = sqrt(pow((vertice_1_x- vertice_2_x), 2) + pow((vertice_1_y - vertice_2_y), 2));
        const float length_pixel_1x = abs(vertice_1_x - pixel_x);
        const float length_pixel_2x = abs(vertice_2_x - pixel_x);
        const float length_pixel_1y = abs(vertice_1_y - pixel_y);
        const float length_pixel_2y = abs(vertice_2_y - pixel_y);
        const float dline_dvx_1 = length_pixel_1x/length_total;
        const float dline_dvy_1 = length_pixel_1y/length_total;
        const float dline_dvx_2 = length_pixel_2x/length_total;
        const float dline_dvy_2 = length_pixel_2y/length_total;
        
        dline_dv_parameter[i*6 + 0] = grad_1;
        dline_dv_parameter[i*6 + 1] = grad_2;
        dline_dv_parameter[i*6 + 2] = dline_dvx_1;
        dline_dv_parameter[i*6 + 3] = dline_dvy_1;
        dline_dv_parameter[i*6 + 4] = dline_dvx_2;
        dline_dv_parameter[i*6 + 5] = dline_dvy_2;

        image_output[i*2 + 0] = pixel_x;
        image_output[i*2 + 1] = pixel_y;
    }

}

} //namespace



class RasterizeLinesOp : public OpKernel {
    public:
        explicit RasterizeLinesOp(OpKernelConstruction* context) : OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("image_width", &image_width_));
            OP_REQUIRES(
                context, image_width_ > 0, 
                InvalidArgument("Image width must be > 0, got", image_width_));

            OP_REQUIRES_OK(context, context->GetAttr("image_height", &image_height_));
            OP_REQUIRES(
                context, image_height_ > 0, 
                InvalidArgument("Image height must be > 0, got", image_height_));
        }
        
        ~RasterizeLinesOp() override {}
        void Compute(OpKernelContext* context) override {
            const Tensor& vertice1_tensor = context->input(0);
            OP_REQUIRES(
                context, 
                PartialTensorShape({2, }).IsCompatibleWith(vertice1_tensor.shape()),
                InvalidArgument("Line points expects to have shape (2, )"));
            auto vertice1_flat = vertice1_tensor.flat<float>();
            const float* vertice_1 = vertice1_flat.data();


            const Tensor& vertice2_tensor = context->input(1);
            OP_REQUIRES(
                context, 
                PartialTensorShape({2, }).IsCompatibleWith(vertice2_tensor.shape()),
                InvalidArgument("Line points expects to have shape (2, )"));
            auto vertice2_flat = vertice2_tensor.flat<float>();
            const float* vertice_2 = vertice2_flat.data(); 
           
  
            Tensor* dline_dv_parameter_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({20, 6}), &dline_dv_parameter_tensor));
            
            Tensor* pixel_count_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({1, }), &pixel_count_tensor));

            Tensor* image_output_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({20, 2}), &image_output_tensor));
            
            // Setting all initial values to zero
            dline_dv_parameter_tensor->flat<float>().setZero();
            pixel_count_tensor->flat<int32>().setZero();
            image_output_tensor->flat<int32>().setZero();

            RasterizeLinesImpl(vertice_1, vertice_2, dline_dv_parameter_tensor->flat<float>().data(), 
                                pixel_count_tensor->flat<int32>().data(), image_output_tensor->flat<int32>().data());
        }

    private:
        TF_DISALLOW_COPY_AND_ASSIGN(RasterizeLinesOp);

        int image_width_;
        int image_height_;
};

REGISTER_KERNEL_BUILDER(Name("RasterizeLines").Device(DEVICE_CPU), RasterizeLinesOp);

}   //namespace tf_line_renderer


