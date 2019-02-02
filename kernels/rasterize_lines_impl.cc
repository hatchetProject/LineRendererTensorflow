#include <algorithm>
#include <cmath>
#include <math.h>
#include <cmath>

#include "rasterize_lines_impl.h"

namespace tf_line_renderer{

namespace {

inline int* DrawBresenhamline(int x0, int y0, int x1, int y1)
{
//Returns the pixel coordinates on the rendered line
//Including the starting point and the ending point
	int dx = x1 - x0;
	int dy = y1 - y0;
	int ux = dx >0 ?1:-1;
	int uy = dx >0 ?1:-1;
	int dx2 = dx <<1;
	int dy2 = dy <<1;
        int* res = new int[200];
        int count = 0;
	if(abs(dx)>abs(dy)){
		int e = -dx;
		int x = x0;
		int y = y0;
		for (x = x0; x < x1;x+=ux)
		{
			//printf ("%d,%d\n",x, y);
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
			//printf ("%d,%d\n",x, y);
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
        int result[2*count+2];
        for (int i = 0; i < count; ++i)
            result[i] = res[i];
        delete []res;
        result[2*count] = x1;
        result[2*count+1] = y1;
        return result;
}



} //namespace




void RasterizeLinesImpl(const float* vertice_1, const float* vertice_2, 
                      int32* pixel_count, float* df_dline){
    int vertice_1_x = vertice_1[0];
    int vertice_1_y = vertice_1[1];
    int vertice_2_x = vertice_2[0];
    int vertice_2_y = vertice_2[1];
    float grad1 = 0, grad2 = 0;
    int* vertices = DrawBresenhamline(vertice_1_x, vertice_1_y, vertice_2_x, vertice_2_y);
    pixel_count[0] = (sizeof(vertices)/sizeof(vertices[0]))/2;
    for (int i = 0; i < pixel_count[0]; ++i){
        const int pixel_x = vertices[i];
        const int pixel_y = vertices[i+1];
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
        for (int j = 0; j < 6; ++j){
              df_dline[i*6 + j] = grad_1;
              df_dline[i*6 + j] = grad_2;
              df_dline[i*6 + j] = dline_dvx_1;
              df_dline[i*6 + j] = dline_dvy_1;
              df_dline[i*6 + j] = dline_dvx_2;
              df_dline[i*6 + j] = dline_dvy_2;
        }
    }

    }

}    //namespace tf_line_renderer






