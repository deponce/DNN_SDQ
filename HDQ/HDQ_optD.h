// HDQ_OptD.h

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

#include <map>
#include <ctime>
#include <chrono>
#include "../Utils/utils.h"
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"
#include <limits>
// #include <algorithm>
// #include <math.h>

using namespace std;
const float MIN_Q_VAL = 1;
class HDQ_OptD{
    public:
        // attributes
        float Q_table_Y[64];
        float Q_table_C[64];
        int seq_len_Y, seq_len_C; // # 8x8 DCT blocks after subsampling
        int n_row;
        int n_col;
        // BLOCK Block;
        int seq_block_dcts[64];
        int DCT_block_shape[3];
        int img_shape_Y[2], img_shape_C[2]; // size of channels after subsampling
        //TODO: P_DC for DC coefficient
        map<int, float> P_DC_Y;
        map<int, float> P_DC_C;
        float J_Y = 10e10;
        float J_C = 10e10;
        int QF_C;
        int QF_Y;
        int J, a, b;
        int QMAX_Y, QMAX_C;
        float DT_Y, DT_C;
        float d_waterlevel_Y, d_waterlevel_C;
        float Loss;
        float EntACY = 0;
        float EntACC = 0;
        float EntDCY = 0;
        float EntDCC = 0;
        float Sen_Map[3][64];
        float MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(float Sen_Map[3][64], int colorspace, int QF_Y, int QF_C, 
                      int J, int a, int b, float DT_Y, float DT_C, 
                      float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C);
        float __call__(vector<vector<vector<float>>>& image);
};

void HDQ_OptD::__init__(float Sen_Map[3][64], int colorspace, int QF_Y, int QF_C, 
                   int J, int a, int b, float DT_Y, float DT_C,
                   float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C){
    minMaxQuantizationStep(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE);
    HDQ_OptD::RSlst.reserve(64);
    HDQ_OptD::IDlst.reserve(64);
    HDQ_OptD::RSlst.clear();
    HDQ_OptD::IDlst.clear();
    quantizationTable(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE, QF_Y, true, HDQ_OptD::Q_table_Y);
    quantizationTable(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE, QF_C, false, HDQ_OptD::Q_table_C);
    HDQ_OptD::J_Y = 10e10;
    HDQ_OptD::J_C = 10e10;
    HDQ_OptD::J = J;
    HDQ_OptD::a = a;
    HDQ_OptD::b = b;
    HDQ_OptD::DT_Y = DT_Y;
    HDQ_OptD::DT_C = DT_C;
    HDQ_OptD::d_waterlevel_Y = d_waterlevel_Y;
    HDQ_OptD::QMAX_Y = QMAX_Y;
    HDQ_OptD::d_waterlevel_C = d_waterlevel_C;
    HDQ_OptD::QMAX_C = QMAX_C;
    for (int i = 0; i <3 ; i++)
    {
        for (int j = 0; j <64 ; j++)
        { 
            HDQ_OptD::Sen_Map[i][j] = Sen_Map[i][j];
        }
    }
}

float HDQ_OptD::__call__(vector<vector<vector<float>>>& image){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, HDQ_OptD::img_shape_Y);
    HDQ_OptD::n_col = HDQ_OptD::img_shape_Y[1];
    HDQ_OptD::n_row = HDQ_OptD::img_shape_Y[0];
    HDQ_OptD::seq_len_Y = pad_shape(HDQ_OptD::img_shape_Y[0], 8)*pad_shape(HDQ_OptD::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(HDQ_OptD::img_shape_Y[0], J);
    int pad_cols = pad_shape(HDQ_OptD::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], HDQ_OptD::img_shape_Y, HDQ_OptD::img_shape_C, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);
    Subsampling(image[2], HDQ_OptD::img_shape_Y, HDQ_OptD::img_shape_C, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);

    HDQ_OptD::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new float[HDQ_OptD::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[HDQ_OptD::seq_len_C][8][8];
    auto blockified_img_Cr = new float[HDQ_OptD::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new float[HDQ_OptD::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new float[HDQ_OptD::seq_len_C][64];
    auto seq_dct_coefs_Cr = new float[HDQ_OptD::seq_len_C][64];

    auto seq_dct_idxs_Y = new float[HDQ_OptD::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[HDQ_OptD::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[HDQ_OptD::seq_len_C][64];

    auto DC_idxs_Y = new float[HDQ_OptD::seq_len_Y];
    auto DC_idxs_Cb = new float[HDQ_OptD::seq_len_C];
    auto DC_idxs_Cr = new float[HDQ_OptD::seq_len_C];

    blockify(image[0], HDQ_OptD::img_shape_Y, blockified_img_Y);
    blockify(image[1], HDQ_OptD::img_shape_C, blockified_img_Cb);    
    blockify(image[2], HDQ_OptD::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, HDQ_OptD::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, HDQ_OptD::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, HDQ_OptD::seq_len_C);


    // // adaptive Qmax 
    // float max_q = std::numeric_limits<float>::min(), tmp, num1, num2;
    // for(int N=0; N<HDQ_OptD::seq_len_C; N++)
    // {
    //     tmp = *max_element(seq_dct_coefs_Y[N], seq_dct_coefs_Y[N] + 64);
    //     if (tmp > max_q) max_q = tmp;
    // }
    // HDQ_OptD::QMAX_Y = 2 * max_q + 1;
    // cout << "Max Quantization Step Y : " <<HDQ_OptD::QMAX_Y << endl;

    // max_q = std::numeric_limits<float>::min();
    // for(int N=0; N<HDQ_OptD::seq_len_C; N++)
    // {
    //     num1 = *max_element(seq_dct_coefs_Cb[N], seq_dct_coefs_Cb[N] + 64);
    //     num2 = *max_element(seq_dct_coefs_Cr[N], seq_dct_coefs_Cr[N] + 64);
    //     tmp = max(num1, num2);
    //     if (tmp > max_q) max_q = tmp;
    // }
    // HDQ_OptD::QMAX_C = 2 * max_q + 1;
    // cout << "Max Quantization Step CbCr : " <<HDQ_OptD::QMAX_C << endl;

    // Customized Quantization Table
    quantizationTable_OptD_Y(HDQ_OptD::Sen_Map, seq_dct_coefs_Y, HDQ_OptD::Q_table_Y, 
                HDQ_OptD::seq_len_Y, HDQ_OptD::DT_Y, HDQ_OptD::d_waterlevel_Y, HDQ_OptD::QMAX_Y);
    // cout << "DT_Y = " << HDQ_OptD::DT_Y << "\t" << "d_waterLevel_Y = " << HDQ_OptD::d_waterlevel_Y << endl;
    quantizationTable_OptD_C(HDQ_OptD::Sen_Map, seq_dct_coefs_Cb, seq_dct_coefs_Cr, HDQ_OptD::Q_table_C
        , HDQ_OptD::seq_len_C, HDQ_OptD::DT_C, HDQ_OptD::d_waterlevel_C, HDQ_OptD::QMAX_C);
    // cout << "DT_C = " << HDQ_OptD::DT_C << "\t" << "d_waterLevel_C = " << HDQ_OptD::d_waterlevel_C << endl;
   


    //

    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             HDQ_OptD::Q_table_Y, HDQ_OptD::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             HDQ_OptD::Q_table_C,HDQ_OptD::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             HDQ_OptD::Q_table_C,HDQ_OptD::seq_len_C);

    
    // cout << "number of blocks: " << seq_len_C << endl;
    // cout << "Cb indices" << endl;
    // for(int N=0; N<HDQ_OptD::seq_len_C; N++)
    // {
    //     cout << seq_dct_idxs_Cb[N][33] << endl;
    // }
    // cout << "Cr indices" << endl;
    // for(int N=0; N<HDQ_OptD::seq_len_C; N++)
    // {
    //     cout << seq_dct_idxs_Cr[N][33] << endl;
    // }
    
    map<int, float> DC_P;
    map<int, float> AC_Y_P;
    map<int, float> AC_C_P;
    DC_P.clear();
    float EntDCY=0;
    float EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, HDQ_OptD::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, HDQ_OptD::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, HDQ_OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, HDQ_OptD::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, HDQ_OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, HDQ_OptD::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    DC_P.clear();

    for(i=0; i<HDQ_OptD::seq_len_Y; i++){
        HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], HDQ_OptD::RSlst, HDQ_OptD::IDlst);
        cal_P_from_RSlst(HDQ_OptD::RSlst, AC_Y_P);
    }
    HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
    AC_Y_P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(AC_Y_P);
    AC_Y_P.clear();

    for(i=0; i<HDQ_OptD::seq_len_C; i++){
        HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], HDQ_OptD::RSlst, HDQ_OptD::IDlst);
        cal_P_from_RSlst(HDQ_OptD::RSlst, AC_C_P);
        HDQ_OptD::RSlst.clear(); HDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], HDQ_OptD::RSlst, HDQ_OptD::IDlst);
        cal_P_from_RSlst(HDQ_OptD::RSlst, AC_C_P);
    }
    AC_C_P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(AC_C_P);
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE; // Run_length coding
    BPP = file_size/HDQ_OptD::img_shape_Y[0]/HDQ_OptD::img_shape_Y[1];
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, HDQ_OptD::Q_table_Y, HDQ_OptD::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, HDQ_OptD::Q_table_Y, HDQ_OptD::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, HDQ_OptD::Q_table_Y, HDQ_OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, HDQ_OptD::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, HDQ_OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, HDQ_OptD::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], HDQ_OptD::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], HDQ_OptD::img_shape_C);
    deblockify(blockified_img_Cr, image[2], HDQ_OptD::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], HDQ_OptD::img_shape_Y, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);
    Upsampling(image[2], HDQ_OptD::img_shape_Y, HDQ_OptD::J, HDQ_OptD::a, HDQ_OptD::b);
    return BPP;
}