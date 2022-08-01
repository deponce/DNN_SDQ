// SDQ_Class.h

// MIT License

// Copyright (c) 2022 deponce(Linfeng Ye), Kaixiang Zheng, University of Waterloo

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

#define DC_OPT 1
#define NUM_ITER 3

#include <map>
#include "block.h"
#include <ctime>
#include <chrono>
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"

using namespace std;

class SDQ{
    public:
        // attributes
        float Beta_S;
        float Beta_W;
        float Beta_X;
        float Q_table_Y[64];
        float Q_table_C[64];
        int seq_len_Y, seq_len_C; // # 8x8 DCT blocks after subsampling
        int n_row;
        int n_col;
        BLOCK Block;
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
        float MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE;
        vector<int> RSlst;
        vector<int> IDlst;
        void __init__(float eps, float Beta_S, float Beta_W, float Beta_X,
                      float Lmbda, float Sen_Map[3][64], int colorspace, int QF_Y, int QF_C, 
                      int J, int a, int b, float DT_Y, float DT_C, 
                      float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C);
        void opt_Q_Y_DC(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]);
        void opt_Q_C_DC(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                        float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]);
        void opt_DC_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]);
        void opt_DC_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64], 
                      float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]);
        void opt_RS_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]);
        void opt_RS_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64],
                      float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]);
        void opt_Q_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]);
        void opt_Q_C(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                     float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]);
        float __call__(vector<vector<vector<float>>>& image);
};

void SDQ::__init__(float eps, float Beta_S, float Beta_W, float Beta_X,
                   float Lmbda, float Sen_Map[3][64], int colorspace, int QF_Y, int QF_C, 
                   int J, int a, int b, float DT_Y, float DT_C,
                   float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C){

    minMaxQuantizationStep(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE);
    SDQ::RSlst.reserve(64);
    SDQ::IDlst.reserve(64);
    SDQ::RSlst.clear();
    SDQ::IDlst.clear();
    SDQ::Beta_S = Beta_S;
    SDQ::Beta_W = Beta_W;
    SDQ::Beta_X = Beta_X;
    quantizationTable(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE, QF_Y, true, SDQ::Q_table_Y);
    quantizationTable(colorspace, MINQVALUE, MAXQVALUE,  QUANTIZATION_SCALE, QF_C, false, SDQ::Q_table_C);
    SDQ::Block.__init__(eps, Beta_S, Beta_W, Beta_X, Lmbda, Sen_Map);
    SDQ::J_Y = 10e10;
    SDQ::J_C = 10e10;
    SDQ::J = J;
    SDQ::a = a;
    SDQ::b = b;
    SDQ::DT_Y = DT_Y;
    SDQ::DT_C = DT_C;
    SDQ::d_waterlevel_Y = d_waterlevel_Y;
    SDQ::QMAX_Y = QMAX_Y;
    SDQ::d_waterlevel_C = d_waterlevel_C;
    SDQ::QMAX_C = QMAX_C;
}

float SDQ::__call__(vector<vector<vector<float>>>& image){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, SDQ::img_shape_Y);
    SDQ::n_col = SDQ::img_shape_Y[1];
    SDQ::n_row = SDQ::img_shape_Y[0];
    SDQ::seq_len_Y = pad_shape(SDQ::img_shape_Y[0], 8)*pad_shape(SDQ::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(SDQ::img_shape_Y[0], J);
    int pad_cols = pad_shape(SDQ::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], SDQ::img_shape_Y, SDQ::img_shape_C, SDQ::J, SDQ::a, SDQ::b);
    Subsampling(image[2], SDQ::img_shape_Y, SDQ::img_shape_C, SDQ::J, SDQ::a, SDQ::b);

    SDQ::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new float[SDQ::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[SDQ::seq_len_C][8][8];
    auto blockified_img_Cr = new float[SDQ::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new float[SDQ::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new float[SDQ::seq_len_C][64];
    auto seq_dct_coefs_Cr = new float[SDQ::seq_len_C][64];

    auto seq_dct_idxs_Y = new float[SDQ::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[SDQ::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[SDQ::seq_len_C][64];

    auto DC_idxs_Y = new float[SDQ::seq_len_Y];
    auto DC_idxs_Cb = new float[SDQ::seq_len_C];
    auto DC_idxs_Cr = new float[SDQ::seq_len_C];

    blockify(image[0], SDQ::img_shape_Y, blockified_img_Y);
    blockify(image[1], SDQ::img_shape_C, blockified_img_Cb);
    blockify(image[2], SDQ::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, SDQ::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, SDQ::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, SDQ::seq_len_C);

    // Customized Quantization Table
    quantizationTable_OptD_Y(seq_dct_coefs_Y, SDQ::Q_table_Y, SDQ::seq_len_Y, SDQ::DT_Y, SDQ::d_waterlevel_Y, SDQ::QMAX_Y);
    cout << "DT_Y = " << SDQ::DT_Y << "\t" << "d_waterLevel_Y = " << SDQ::d_waterlevel_Y << endl;
    // quantizationTable_OptD_C(seq_dct_coefs_Cb, seq_dct_coefs_Cr, SDQ::Q_table_C, SDQ::seq_len_C, SDQ::DT_C, SDQ::d_waterlevel_C, SDQ::QMAX_C);

    // [LENA] Just to check seq_dct_coefs_Cb = seq_dct_coefs_Cr = seq_dct_coefs_Y 
    // quantizationTable_OptD_C(seq_dct_coefs_Y, seq_dct_coefs_Y, SDQ::Q_table_C, SDQ::seq_len_C, SDQ::DT_C, SDQ::d_waterlevel_C, SDQ::QMAX_C);
    // cout << "DT_C = " << SDQ::DT_C << "\t" << "d_waterLevel_C = " << SDQ::d_waterlevel_C << endl;
    
    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             SDQ::Q_table_Y, SDQ::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             SDQ::Q_table_C,SDQ::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             SDQ::Q_table_C,SDQ::seq_len_C);
    
/////////////////////////////////////////////////////////////////////////////
#if DC_OPT > 0
    // DC optimization
    // Y channel
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, P_DC_Y, seq_len_Y);  // initialize P_DC_Y
    
    for(int i=0; i<NUM_ITER; i++)
    {
        opt_DC_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);       // optimize seq_dct_idxs_Y
        opt_Q_Y_DC(seq_dct_idxs_Y,seq_dct_coefs_Y);     // update Q_table_Y[0]
        P_DC_Y.clear();
        DPCM(seq_dct_idxs_Y, DC_idxs_Y, seq_len_Y);
        cal_P_from_DIFF(DC_idxs_Y, P_DC_Y, seq_len_Y);  // update P_DC_Y
    }
    EntDCY = calHuffmanCodeSize(P_DC_Y);            // cal huffman size
    // Cb Cr channels
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, P_DC_C, seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, P_DC_C, seq_len_C); // initialize P_DC_C
    for(int i=0; i<NUM_ITER; i++)
    {
        opt_DC_C(seq_dct_idxs_Cr, seq_dct_coefs_Cr, 
                 seq_dct_idxs_Cb, seq_dct_coefs_Cb);    // optimize seq_dct_idxs_Cb/Cr
        opt_Q_C_DC(seq_dct_idxs_Cr, seq_dct_coefs_Cr, 
                   seq_dct_idxs_Cb, seq_dct_coefs_Cb);  // update Q_table_C[0]
        P_DC_C.clear();
        DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, seq_len_C);
        cal_P_from_DIFF(DC_idxs_Cb, P_DC_C, seq_len_C);
        DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, seq_len_C);
        cal_P_from_DIFF(DC_idxs_Cr, P_DC_C, seq_len_C); // update P_DC_C
    }
    EntDCC = calHuffmanCodeSize(P_DC_C);            // cal huffman size

///////////////////////////////////////////////////////////////////////////
#else
    //without DC opt
    map<int, float> DC_P;
    DC_P.clear();
    float EntDCY=0;
    float EntDCC=0;
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, SDQ::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, SDQ::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    // cout<<"EntDCY:"<<EntDCY<<endl;
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, SDQ::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, SDQ::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, SDQ::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, SDQ::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    // cout<<"EntDCC:"<<EntDCC<<endl;
    DC_P.clear();

#endif
/////////////////////////////////////////////////


    // AC optimization
    // Y channel
    for(i=0; i<NUM_ITER; i++){
        SDQ::Loss = 0;
        SDQ::Block.state.ent=0;
        SDQ::opt_RS_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);
        SDQ::opt_Q_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);
        // std::cout<<SDQ::Loss<<std::endl;
    }
    SDQ::Block.P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(SDQ::Block.P);      // cal huffman size
    SDQ::Block.P.clear();
    // Cb Cr channels
    for(i=0; i<NUM_ITER; i++){
        SDQ::Loss = 0;
        SDQ::Block.state.ent=0;
        SDQ::opt_RS_C(seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                      seq_dct_idxs_Cr,seq_dct_coefs_Cr);
        SDQ::opt_Q_C(seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                     seq_dct_idxs_Cr,seq_dct_coefs_Cr);
        // std::cout<<SDQ::Loss<<std::endl;
    }
    SDQ::Block.P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(SDQ::Block.P);      // cal huffman size
    
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE; // Run_length coding
    BPP = file_size/SDQ::img_shape_Y[0]/SDQ::img_shape_Y[1];
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, SDQ::Q_table_Y, SDQ::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, SDQ::Q_table_C, SDQ::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, SDQ::Q_table_C, SDQ::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, SDQ::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, SDQ::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, SDQ::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], SDQ::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], SDQ::img_shape_C);
    deblockify(blockified_img_Cr, image[2], SDQ::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], SDQ::img_shape_Y, SDQ::J, SDQ::a, SDQ::b);
    Upsampling(image[2], SDQ::img_shape_Y, SDQ::J, SDQ::a, SDQ::b);
    return BPP;
}
