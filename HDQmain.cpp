#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include "./HDQ/HDQ.h"

using namespace cv;
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
  double Beta = 1e9;
  double Lmbda = 1e9;
  double eps = 0.1;
  while ((option = getopt(argc, argv, "hiM:P:J:a:b:e:Q:q:B:e:L:")) !=-1){
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
        << "-e #: the threshold "<<std::endl<<std::flush;
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
    <<"eps: "<<eps<<std::endl<<std::flush;
  }
  cv::Mat image;
  image = cv::imread(IM_PATH);
  if(! image.data){
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
      }
  double Beta_S = Beta;
  double Beta_W = Beta;
  double Beta_X = Beta;


  
  double W_rgb2swx[3][3];
  double W_swx2rgb[3][3];
  double bias_rgb2swx = 128;
  int nrows = image.rows;
  int ncols = image.cols;
  vector<vector<vector<double>>> Vect_img(3, vector<vector<double>>(nrows, vector<double>(ncols, 0)));
  vector<vector<vector<double>>> ori_img(3, vector<vector<double>>(nrows, vector<double>(ncols, 0)));
  Mat2Vector(image, Vect_img);
  Mat2Vector(image, ori_img);
  // cout<<Model;
    
  rgb2YUV(Vect_img, W_rgb2swx, bias_rgb2swx);
  // rgb2swx(Vect_img, W_rgb2swx, bias_rgb2swx);
  
  HDQ hdq;
  hdq.__init__(eps, Beta_S, Beta_W, Beta_X,
               Lmbda, QF_Y , QF_C, 
               J, a, b);
  double BPP = hdq.__call__(Vect_img); //Vect_img is the compressed dequantilzed image after sdq.__call__()
  cout<<"BPP: "<<BPP<<endl;

  YUV2rgb(Vect_img, W_rgb2swx, bias_rgb2swx);

  // swx2rgb(Vect_img, W_swx2rgb, bias_rgb2swx);
  double psnrVal = PSNRY(Vect_img, ori_img);
  cout<<"PSNR: "<<psnrVal<<endl;

  Vector2Mat(Vect_img, image);
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window",  image);
  cv::waitKey(0);
  return 0;
}
