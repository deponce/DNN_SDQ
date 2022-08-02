// SDQmain.cpp

// MIT License

// Copyright (c) 2022 Ahmed Hussein Salamah, Kaixiang Zheng, deponce(Linfeng Ye), University of Waterloo

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

// #include <opencv2/highgui.hpp>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./SDQ/SDQ.h"
#include "./SDQ/load.h"

// using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  int option;
  char* IM_PATH; //Resnet18, Squeezenet, Alexnet, VGG11
  char* Model;
  bool arginfo = false;
  int J = 4;
  int a = 4;
  int b = 4;
  int QF_Y = 20;
  int QF_C = 30;
  float Beta = 1e9;
  float Lmbda = 1e9;
  float eps = 0.1;
  int Qmax_Y = 46, Qmax_C = 255;
  float d_waterlevel_Y = 0, d_waterlevel_C = 0;  
  float DT_Y = 0, DT_C = 0;
  float BPP_t = 1e16, Lmbda_step;
while ((option = getopt(argc, argv, "hiM:P:J:a:b:e:Q:q:B:L:e:T:t:D:d:X:x:U:")) !=-1){
    switch (option){
      case 'h':{
        std::cout<< "-i: output the model's name and path of the input image"<<std::endl<<std::flush
        << "-M #: model's name"<<std::endl
        << "-P #: path to input image"<<std::endl
        << "-J #: subsampling mode"<<std::endl
        << "-a #: subsampling mode"<<std::endl
        << "-b #: subsampling mode"<<std::endl
        << "-Q #: intial quality factor of the quantization table for Y channel"<<std::endl
        << "-q #: intial quality factor of the quantization table for C channel"<<std::endl
        << "-B #: the 1st Langrangian factor"<<std::endl
        << "-L #: the 2nd Langrangian factor"<<std::endl
        << "-e #: SDQ threshold "<<std::endl
        << "-T #: Dist Target for Y "<<std::endl
        << "-t #: Dist Target for C "<<std::endl
        << "-D #: Dist Waterlevel for Y "<<std::endl
        << "-d #: Dist Waterlevel for C"<<std::endl
        << "-X #: Max OptD Quantization step Y "<<std::endl
        << "-x #: Max OptD Quantization step C"<<std::endl
        << "-U #: Target BPP"<<std::endl
        <<std::flush;
        break;}
      case 'i':{arginfo = true; break;}
      case 'M':{Model = optarg; break;}
      case 'P':{IM_PATH = optarg; break;}
      case 'J':{J = atoi(optarg); break;}
      case 'a':{a = atoi(optarg); break;}
      case 'b':{b = atoi(optarg); break;}
      case 'Q':{QF_Y = atoi(optarg); break;}
      case 'q':{QF_C = atoi(optarg); break;}
      case 'B':{Beta = atof(optarg); break;}
      case 'L':{Lmbda = atof(optarg); break;}
      case 'e':{eps = atof(optarg); break;}
      case 'T':{DT_Y = atof(optarg); break;}
      case 't':{DT_C = atof(optarg); break;}
      case 'D':{d_waterlevel_Y = atof(optarg); break;}
      case 'd':{d_waterlevel_C = atof(optarg); break;}
      case 'X':{Qmax_Y = atoi(optarg); break;}
      case 'x':{Qmax_C = atoi(optarg); break;}
      case 'U':{BPP_t = atof(optarg); break;}
      default: {break;}
    }
  }
  if(arginfo){
    std::cout<<std::endl
    <<"Model: "<<Model<<std::endl
    <<"Image path: "<<IM_PATH<<std::endl
    <<"Beta: "<<Beta<<std::endl
    <<"J,a,b: "<<J<<" "<<a<<" "<<b<<std::endl
    <<"QF_Y:  "<<QF_Y<<std::endl
    <<"QF_C:  "<<QF_C<<std::endl
    <<"Dist Target for Y, C: "<< DT_Y << " , "<< DT_C <<std::endl
    <<"Dist Waterlevel for Y, C: "<< d_waterlevel_Y << " , "<< d_waterlevel_C <<std::endl
    <<"Max OptD Quantization step Y, C: "<< Qmax_Y << " , "<< Qmax_C <<std::endl
    <<"eps: "<<eps<<std::endl
    <<std::flush;
  }
  cv::Mat image;
  image = cv::imread(IM_PATH);
  if(! image.data){
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
      }
  float Beta_S = Beta;
  float Beta_W = Beta;
  float Beta_X = Beta;
  float Sen_Map[3][64]={0};
  int colorspace = 0;
  float psnrVal = 0, psnrVal_t, BPP; 
  LoadSenMap(Model, Sen_Map);
  
  float W_rgb2swx[3][3];
  float W_swx2rgb[3][3];
  LoadColorConvW(Model, W_rgb2swx, W_swx2rgb);
  float bias_rgb2swx = 128;
  int nrows = image.rows;
  int ncols = image.cols;
  vector<vector<vector<float>>> Vect_img(3, vector<vector<float>>(nrows, vector<float>(ncols, 0)));
  vector<vector<vector<float>>> ori_img(3, vector<vector<float>>(nrows, vector<float>(ncols, 0)));
  Mat2Vector(image, Vect_img);
  Mat2Vector(image, ori_img);


  int count = 0;
  Lmbda_step = 0.1;
  bool condition;
  do
  {
    rgb2YUV(Vect_img);
    // rgb2YYY(Vect_img);
    // rgb2swx(Vect_img, W_rgb2swx, bias_rgb2swx);

    SDQ sdq;
    sdq.__init__(eps, Beta_S, Beta_W, Beta_X,
                 Lmbda, Sen_Map, colorspace, QF_Y , QF_C, 
                 J, a, b, DT_Y, DT_C, d_waterlevel_Y, d_waterlevel_C, Qmax_Y, Qmax_C);
    BPP = sdq.__call__(Vect_img); //Vect_img is the compressed dequantilzed image after sdq.__call__()

    YUV2rgb(Vect_img);
    // YYY2rgb(Vect_img);

    // swx2rgb(Vect_img, W_swx2rgb, bias_rgb2swx);
    psnrVal = PSNRY(Vect_img, ori_img);

    cout<<"Lmbda: "<<Lmbda<<endl;
    cout<<"BPP: "<< BPP <<endl;
    cout<<"PSNR: "<<psnrVal<<endl<<endl;
    
    if ((BPP - BPP_t) > 0)
    {
      Lmbda = Lmbda - Lmbda_step;
    }
    else if ((BPP - BPP_t) < 0)
    { 
      Lmbda = Lmbda + Lmbda_step;
    }
    count++;
    condition = ((BPP - BPP_t) < 0.03) && (count < 10);
    if (condition) Vect_img = ori_img; 
  }
  while(condition);

  // Vector2Mat(Vect_img, image);
  // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  // cv::imshow("Display window",  image);
  // cv::waitKey(0);
  return 0;
}
