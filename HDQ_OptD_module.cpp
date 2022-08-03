// HDQ_module.cpp

// MIT License

// Copyright (c) 2022 deponce(Linfeng Ye), University of Waterloo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <opencv2/highgui.hpp>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./HDQ/HDQ_optD.h"
#include "./SDQ/load.h"
// using namespace cv;
using namespace std;

// -------------
// pure C++ code
// -------------
using namespace std;

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
std::pair<py::array, float> py__call__(py::array_t<float, py::array::c_style | py::array::forcecast> array,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> SenMap,
                                       string Model, int colorspace, int J, int a, int b, 
                                       float DT_Y, float DT_C, float d_waterlevel_Y, float d_waterlevel_C, 
                                       int Qmax_Y, int Qmax_C){
  unsigned long size[2];
  size[0] = (unsigned long)array.shape()[1];
  size[1] = (unsigned long)array.shape()[2];
  // allocate std::vector (to pass to the C++ function)
  vector<float> pos(array.size());
  vector<vector<vector<float>>> Vect_img(3, vector<vector<float>>(size[0], vector<float>(size[1], 0)));
  // copy py::array -> std::vector
  memcpy(pos.data(), array.data(),array.size()*sizeof(float));
  // call pure C++ function
  //TODO::
  float BPP =0;
  vector<float> result(array.size());
  seq2img(pos, Vect_img, size[0], size[1]);
  float Sen_Map[3][64]={0};
  memcpy(Sen_Map, SenMap.data(), 3*64*sizeof(float));

  float W_rgb2swx[3][3];
  float W_swx2rgb[3][3];
  float biasPerImage[3] = {0,0,0};
  float bias_rgb2swx = 128.;

  int QF_Y = 50;
  int QF_C = 50;


  // string Model = "Alexnet";
  // LoadColorConvW(Model, W_rgb2swx, W_swx2rgb);
  // rgb2swx(Vect_img, W_rgb2swx, bias_rgb2swx);
  // rgb2YUV(Vect_img);

  
  if(colorspace == 0)
  {
    rgb2YUV(Vect_img);
  }
  else if(colorspace == 1)
  {
    LoadColorConvW(Model, W_rgb2swx, W_swx2rgb);
    rgb2swx(Vect_img, W_rgb2swx, bias_rgb2swx);
  }
  else if(colorspace >= 2)
  {
    // remove the mean for each image
    LoadColorConvW(Model, W_rgb2swx, W_swx2rgb);
    rgb2swx_PerImage(Vect_img, W_rgb2swx, biasPerImage);
  }


  HDQ_OptD hdq;
  hdq.__init__(Sen_Map, colorspace, QF_Y , QF_C, J, a, b, DT_Y, DT_C, d_waterlevel_Y, d_waterlevel_C, Qmax_Y, Qmax_C);
  BPP = hdq.__call__(Vect_img); // Vect_img is the compressed dequantilzed image after sdq.__call__()

  // YUV2rgb(Vect_img);
  // swx2rgb(Vect_img, W_swx2rgb, bias_rgb2swx);

  
  if(colorspace == 0)
  {
    YUV2rgb(Vect_img);
  }
  else if(colorspace == 1)
  {
    swx2rgb(Vect_img, W_swx2rgb, bias_rgb2swx);
  }
  else if(colorspace >= 2)
  {
    swx2rgb_PerImage(Vect_img, W_swx2rgb, biasPerImage);
  }


  img2seq(Vect_img, result, size[0], size[1]);
  int ndim = 3;
  vector<unsigned long> shape   = { 3, size[0], size[1]};
  vector<unsigned long> strides = { size[0]*size[1]*sizeof(float),
                                    size[1]*sizeof(float), sizeof(float)};
  // delete [] Sen_Map;
  // return 2-D NumPy array
  return std::make_pair(py::array(py::buffer_info(
    result.data(),                          /* data as contiguous array */
    sizeof(float),                          /* size of one scalar       */
    py::format_descriptor<float>::format(), /* data type                */
    ndim,                                    /* number of dimensions     */
    shape,                                   /* shape of the matrix      */
    strides                                  /* strides for each axis    */
  )), BPP);
}
// wrap as Python module
PYBIND11_MODULE(HDQ_OptD,m)
{
  m.doc() = "HDQ_OptD API";
  m.def("__call__", &py__call__, py::return_value_policy::move ,"Calculate the length of an array of vectors");
}
