#ifndef LINE_RENDERER_KERNELS_RASTERIZE_LINES_IMPL_H_
#define LINE_RENDERER_KERNELS_RASTERIZE_LINES_IMPL_H_

namespace tf_line_renderer{

typedef int int32;
typedef long long int64;



void RasterizeLinesImpl(const float* vertice_1, const float* vertice_2, 
                      int32* pixel_count, float* df_dline);
} // namespace tf_line_renderer


#endif  // LINE_RENDERER_KERNELS_RASTERIZE_LINES_IMPL_H_




