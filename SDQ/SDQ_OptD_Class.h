// SDQ_Class.h

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

#define DC_OPT 1
#define NUM_ITER 3

#include <map>
#include "block.h"
#include <ctime>
#include <chrono>
#include "../Utils/Q_Table.h"
#include "../EntCoding/Huffman.h"

using namespace std;

class SDQ_OptD{
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
        float (*Sen_Map)[64];
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

void SDQ_OptD::__init__(float eps, float Beta_S, float Beta_W, float Beta_X,
                   float Lmbda, float Sen_Map[3][64], int colorspace, int QF_Y, int QF_C, 
                   int J, int a, int b, float DT_Y, float DT_C,
                   float d_waterlevel_Y, float d_waterlevel_C, int QMAX_Y, int QMAX_C){

    minMaxQuantizationStep(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE);
    SDQ_OptD::RSlst.reserve(64);
    SDQ_OptD::IDlst.reserve(64);
    SDQ_OptD::RSlst.clear();
    SDQ_OptD::IDlst.clear();
    SDQ_OptD::Beta_S = Beta_S;
    SDQ_OptD::Beta_W = Beta_W;
    SDQ_OptD::Beta_X = Beta_X;
    quantizationTable(colorspace, MINQVALUE, MAXQVALUE, QUANTIZATION_SCALE, QF_Y, true, SDQ_OptD::Q_table_Y);
    quantizationTable(colorspace, MINQVALUE, MAXQVALUE,  QUANTIZATION_SCALE, QF_C, false, SDQ_OptD::Q_table_C);
    SDQ_OptD::Block.__init__(eps, Beta_S, Beta_W, Beta_X, Lmbda, Sen_Map);
    SDQ_OptD::J_Y = 10e10;
    SDQ_OptD::J_C = 10e10;
    SDQ_OptD::J = J;
    SDQ_OptD::a = a;
    SDQ_OptD::b = b;
    SDQ_OptD::DT_Y = DT_Y;
    SDQ_OptD::DT_C = DT_C;
    SDQ_OptD::d_waterlevel_Y = d_waterlevel_Y;
    SDQ_OptD::QMAX_Y = QMAX_Y;
    SDQ_OptD::d_waterlevel_C = d_waterlevel_C;
    SDQ_OptD::QMAX_C = QMAX_C;
    SDQ_OptD::Sen_Map = Sen_Map;
}

float SDQ_OptD::__call__(vector<vector<vector<float>>>& image){
    int i,j,k,l,r,s;
    int BITS[33];
    int size;
    std::fill_n(BITS, 33, 0);
    shape(image, SDQ_OptD::img_shape_Y);
    SDQ_OptD::n_col = SDQ_OptD::img_shape_Y[1];
    SDQ_OptD::n_row = SDQ_OptD::img_shape_Y[0];
    SDQ_OptD::seq_len_Y = pad_shape(SDQ_OptD::img_shape_Y[0], 8)*pad_shape(SDQ_OptD::img_shape_Y[1], 8)/64;
    int SmplHstep = floor(J/a);
    int SmplVstep;
    if(b==0){SmplVstep = 2;} else{SmplVstep = 1;}
    int pad_rows = pad_shape(SDQ_OptD::img_shape_Y[0], J);
    int pad_cols = pad_shape(SDQ_OptD::img_shape_Y[1], J);
    int Smplrows = pad_rows/SmplVstep;
    int Smplcols = pad_cols/SmplHstep;
    Subsampling(image[1], SDQ_OptD::img_shape_Y, SDQ_OptD::img_shape_C, SDQ_OptD::J, SDQ_OptD::a, SDQ_OptD::b);
    Subsampling(image[2], SDQ_OptD::img_shape_Y, SDQ_OptD::img_shape_C, SDQ_OptD::J, SDQ_OptD::a, SDQ_OptD::b);

    SDQ_OptD::seq_len_C = pad_shape(Smplcols, 8)*pad_shape(Smplrows, 8)/64;
    auto blockified_img_Y = new float[SDQ_OptD::seq_len_Y][8][8];
    auto blockified_img_Cb = new float[SDQ_OptD::seq_len_C][8][8];
    auto blockified_img_Cr = new float[SDQ_OptD::seq_len_C][8][8];

    auto seq_dct_coefs_Y = new float[SDQ_OptD::seq_len_Y][64];
    auto seq_dct_coefs_Cb = new float[SDQ_OptD::seq_len_C][64];
    auto seq_dct_coefs_Cr = new float[SDQ_OptD::seq_len_C][64];

    auto seq_dct_idxs_Y = new float[SDQ_OptD::seq_len_Y][64];
    auto seq_dct_idxs_Cb = new float[SDQ_OptD::seq_len_C][64];
    auto seq_dct_idxs_Cr = new float[SDQ_OptD::seq_len_C][64];

    auto DC_idxs_Y = new float[SDQ_OptD::seq_len_Y];
    auto DC_idxs_Cb = new float[SDQ_OptD::seq_len_C];
    auto DC_idxs_Cr = new float[SDQ_OptD::seq_len_C];

    blockify(image[0], SDQ_OptD::img_shape_Y, blockified_img_Y);
    blockify(image[1], SDQ_OptD::img_shape_C, blockified_img_Cb);
    blockify(image[2], SDQ_OptD::img_shape_C, blockified_img_Cr);

    block_2_seqdct(blockified_img_Y, seq_dct_coefs_Y, SDQ_OptD::seq_len_Y);
    block_2_seqdct(blockified_img_Cb, seq_dct_coefs_Cb, SDQ_OptD::seq_len_C);
    block_2_seqdct(blockified_img_Cr, seq_dct_coefs_Cr, SDQ_OptD::seq_len_C);

    // Customized Quantization Table
    quantizationTable_OptD_Y(SDQ_OptD::Sen_Map, seq_dct_coefs_Y, SDQ_OptD::Q_table_Y, 
                SDQ_OptD::seq_len_Y, SDQ_OptD::DT_Y, SDQ_OptD::d_waterlevel_Y, SDQ_OptD::QMAX_Y);
    // cout << "DT_Y = " << SDQ_OptD::DT_Y << "\t" << "d_waterLevel_Y = " << SDQ_OptD::d_waterlevel_Y << endl;
    
    quantizationTable_OptD_C(SDQ_OptD::Sen_Map, seq_dct_coefs_Cb, seq_dct_coefs_Cr, SDQ_OptD::Q_table_C
        , SDQ_OptD::seq_len_C, SDQ_OptD::DT_C, SDQ_OptD::d_waterlevel_C, SDQ_OptD::QMAX_C);
// cout << "DT_C = " << SDQ_OptD::DT_C << "\t" << "d_waterLevel_C = " << SDQ_OptD::d_waterlevel_C << endl;
   
    // [LENA] Just to check seq_dct_coefs_Cb = seq_dct_coefs_Cr = seq_dct_coefs_Y 
    // quantizationTable_OptD_C(seq_dct_coefs_Y, seq_dct_coefs_Y, SDQ_OptD::Q_table_C, SDQ_OptD::seq_len_C, SDQ_OptD::DT_C, SDQ_OptD::d_waterlevel_C, SDQ_OptD::QMAX_C);
    // cout << "DT_C = " << SDQ_OptD::DT_C << "\t" << "d_waterLevel_C = " << SDQ_OptD::d_waterlevel_C << endl;
    
    Quantize(seq_dct_coefs_Y,seq_dct_idxs_Y, 
             SDQ_OptD::Q_table_Y, SDQ_OptD::seq_len_Y);
    Quantize(seq_dct_coefs_Cb,seq_dct_idxs_Cb,
             SDQ_OptD::Q_table_C,SDQ_OptD::seq_len_C);
    Quantize(seq_dct_coefs_Cr,seq_dct_idxs_Cr,
             SDQ_OptD::Q_table_C,SDQ_OptD::seq_len_C);
    
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
    DPCM(seq_dct_idxs_Y, DC_idxs_Y, SDQ_OptD::seq_len_Y);
    cal_P_from_DIFF(DC_idxs_Y, DC_P, SDQ_OptD::seq_len_Y);
    DC_P.erase(TOTAL_KEY);
    EntDCY = calHuffmanCodeSize(DC_P);
    // cout<<"EntDCY:"<<EntDCY<<endl;
    DC_P.clear();
    DPCM(seq_dct_idxs_Cb, DC_idxs_Cb, SDQ_OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cb, DC_P, SDQ_OptD::seq_len_C);
    DPCM(seq_dct_idxs_Cr, DC_idxs_Cr, SDQ_OptD::seq_len_C);
    cal_P_from_DIFF(DC_idxs_Cr, DC_P, SDQ_OptD::seq_len_C);
    DC_P.erase(TOTAL_KEY);
    EntDCC = calHuffmanCodeSize(DC_P);
    // cout<<"EntDCC:"<<EntDCC<<endl;
    DC_P.clear();

#endif
/////////////////////////////////////////////////


    // AC optimization
    // Y channel
    for(i=0; i<NUM_ITER; i++){
        SDQ_OptD::Loss = 0;
        SDQ_OptD::Block.state.ent=0;
        SDQ_OptD::opt_RS_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);
        SDQ_OptD::opt_Q_Y(seq_dct_idxs_Y,seq_dct_coefs_Y);
        // std::cout<<SDQ_OptD::Loss<<std::endl;
    }
    SDQ_OptD::Block.P.erase(TOTAL_KEY);
    EntACY = calHuffmanCodeSize(SDQ_OptD::Block.P);      // cal huffman size
    SDQ_OptD::Block.P.clear();
    // Cb Cr channels
    for(i=0; i<NUM_ITER; i++){
        SDQ_OptD::Loss = 0;
        SDQ_OptD::Block.state.ent=0;
        SDQ_OptD::opt_RS_C(seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                      seq_dct_idxs_Cr,seq_dct_coefs_Cr);
        SDQ_OptD::opt_Q_C(seq_dct_idxs_Cb,seq_dct_coefs_Cb,
                     seq_dct_idxs_Cr,seq_dct_coefs_Cr);
        // std::cout<<SDQ_OptD::Loss<<std::endl;
    }
    SDQ_OptD::Block.P.erase(TOTAL_KEY);
    EntACC = calHuffmanCodeSize(SDQ_OptD::Block.P);      // cal huffman size
    
    float BPP=0;
    float file_size = EntACC+EntACY+EntDCC+EntDCY+FLAG_SIZE; // Run_length coding
    BPP = file_size/SDQ_OptD::img_shape_Y[0]/SDQ_OptD::img_shape_Y[1];
    delete [] seq_dct_coefs_Y; delete [] seq_dct_coefs_Cb; delete [] seq_dct_coefs_Cr;
    Dequantize(seq_dct_idxs_Y, SDQ_OptD::Q_table_Y, SDQ_OptD::seq_len_Y); //seq_dct_idxs_Y: [][64]
    Dequantize(seq_dct_idxs_Cb, SDQ_OptD::Q_table_C, SDQ_OptD::seq_len_C);
    Dequantize(seq_dct_idxs_Cr, SDQ_OptD::Q_table_C, SDQ_OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Y, blockified_img_Y, SDQ_OptD::seq_len_Y); //seq_dct_idxs_Y: [][8[8]
    seq_2_blockidct(seq_dct_idxs_Cb, blockified_img_Cb, SDQ_OptD::seq_len_C);
    seq_2_blockidct(seq_dct_idxs_Cr, blockified_img_Cr, SDQ_OptD::seq_len_C);
    delete [] seq_dct_idxs_Y; delete [] seq_dct_idxs_Cb; delete [] seq_dct_idxs_Cr;
    deblockify(blockified_img_Y,  image[0], SDQ_OptD::img_shape_Y); //seq_dct_idxs_Y: [][], crop here!
    deblockify(blockified_img_Cb, image[1], SDQ_OptD::img_shape_C);
    deblockify(blockified_img_Cr, image[2], SDQ_OptD::img_shape_C);
    delete [] blockified_img_Y; delete [] blockified_img_Cb; delete [] blockified_img_Cr;
    Upsampling(image[1], SDQ_OptD::img_shape_Y, SDQ_OptD::J, SDQ_OptD::a, SDQ_OptD::b);
    Upsampling(image[2], SDQ_OptD::img_shape_Y, SDQ_OptD::J, SDQ_OptD::a, SDQ_OptD::b);
    return BPP;
}



void SDQ_OptD::opt_RS_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    int i;
    // C channel
    std::fill_n(SDQ_OptD::Block.state.ID, 64,0);
    std::fill_n(SDQ_OptD::Block.state.rs, 64,0);
    SDQ_OptD::Block.ent.clear();
    SDQ_OptD::Block.P.clear();
    // initialize Py0
    for(i=0; i<SDQ_OptD::seq_len_Y; i++){
        SDQ_OptD::RSlst.clear(); SDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Y[i], SDQ_OptD::RSlst, SDQ_OptD::IDlst);
        cal_P_from_RSlst(SDQ_OptD::RSlst, SDQ_OptD::Block.P);
    }
    SDQ_OptD::RSlst.clear(); SDQ_OptD::IDlst.clear();
    norm(SDQ_OptD::Block.P, SDQ_OptD::Block.ent);
    cal_ent(SDQ_OptD::Block.ent);
    SDQ_OptD::Block.set_channel('S');
    SDQ_OptD::Block.set_Q_table(SDQ_OptD::Q_table_Y); 
    SDQ_OptD::Loss = 0;
    SDQ_OptD::Block.state.ent=0;
    for(i=0; i<SDQ_OptD::seq_len_Y; i++){
        std::fill_n(SDQ_OptD::Block.state.ID, 64,0);
        std::fill_n(SDQ_OptD::Block.state.rs, 64,0);
        SDQ_OptD::Block.cal_RS(seq_dct_coefs_Y[i],
                          seq_dct_idxs_Y[i],
                          SDQ_OptD::RSlst, SDQ_OptD::IDlst);
        RSlst_to_Block(seq_dct_idxs_Y[i][0], SDQ_OptD::RSlst,
                            SDQ_OptD::IDlst, seq_dct_idxs_Y[i]);
        SDQ_OptD::RSlst.clear();SDQ_OptD::IDlst.clear();
        SDQ_OptD::Loss += SDQ_OptD::Block.J;
    }
}


void SDQ_OptD::opt_RS_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64],
                   float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    int i;
    // C channel
    SDQ_OptD::Block.ent.clear();
    SDQ_OptD::Block.P.clear();
    std::fill_n(SDQ_OptD::Block.state.ID, 64,0);
    std::fill_n(SDQ_OptD::Block.state.rs, 64,0);
    // initialize Pc0
    for(i=0; i<SDQ_OptD::seq_len_C; i++){
        SDQ_OptD::RSlst.clear(); SDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cr[i], SDQ_OptD::RSlst, SDQ_OptD::IDlst);
        cal_P_from_RSlst(SDQ_OptD::RSlst, SDQ_OptD::Block.ent);
        SDQ_OptD::RSlst.clear(); SDQ_OptD::IDlst.clear();
        Block_to_RSlst(seq_dct_idxs_Cb[i], SDQ_OptD::RSlst, SDQ_OptD::IDlst);
        cal_P_from_RSlst(SDQ_OptD::RSlst, SDQ_OptD::Block.P);
    }
    SDQ_OptD::RSlst.clear(); SDQ_OptD::IDlst.clear();
    norm(SDQ_OptD::Block.P, SDQ_OptD::Block.ent);
    cal_ent(SDQ_OptD::Block.ent);
    SDQ_OptD::Block.set_channel('W');
    SDQ_OptD::Block.set_Q_table(SDQ_OptD::Q_table_C);
    SDQ_OptD::Loss = 0;
    SDQ_OptD::Block.state.ent=0;
    for(i=0; i<SDQ_OptD::seq_len_C; i++){
        std::fill_n(SDQ_OptD::Block.state.ID, 64,0);
        std::fill_n(SDQ_OptD::Block.state.rs, 64,0);
        SDQ_OptD::Block.cal_RS(seq_dct_coefs_Cb[i],
                          seq_dct_idxs_Cb[i],
                          SDQ_OptD::RSlst, SDQ_OptD::IDlst);
        RSlst_to_Block(seq_dct_idxs_Cb[i][0], SDQ_OptD::RSlst,
                            SDQ_OptD::IDlst, seq_dct_idxs_Cb[i]);
        SDQ_OptD::RSlst.clear();SDQ_OptD::IDlst.clear();
        SDQ_OptD::Loss += SDQ_OptD::Block.J;
    }
    SDQ_OptD::Block.set_channel('X');
    for(i=0; i<SDQ_OptD::seq_len_C; i++){
        std::fill_n(SDQ_OptD::Block.state.ID, 64,0);
        std::fill_n(SDQ_OptD::Block.state.rs, 64,0);
        SDQ_OptD::Block.cal_RS(seq_dct_coefs_Cr[i],
                          seq_dct_idxs_Cr[i],
                          SDQ_OptD::RSlst, SDQ_OptD::IDlst);
        RSlst_to_Block(seq_dct_idxs_Cr[i][0], SDQ_OptD::RSlst,
                            SDQ_OptD::IDlst, seq_dct_idxs_Cr[i]);
        SDQ_OptD::RSlst.clear(); SDQ_OptD::IDlst.clear();
        SDQ_OptD::Loss += SDQ_OptD::Block.J;  
    }
}

void SDQ_OptD::opt_Q_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    float divisor=0;
    float denominator=0;
    float val;
    int i,j;
    //TODO: start with 1
    for(j=1; j<64; j++){
        for(i=0; i<SDQ_OptD::seq_len_Y; i++){
            divisor += seq_dct_coefs_Y[i][j]*seq_dct_idxs_Y[i][j];
            denominator += pow(seq_dct_idxs_Y[i][j],2);
        }
        if(denominator != 0){
            val = divisor/denominator;
            val = round(MinMaxClip(val, MINQVALUE, MAXQVALUE));
            SDQ_OptD::Q_table_Y[j] = val;
        }
        else    Q_table_Y[j] = MAXQVALUE;
        divisor=0; denominator=0; 
    }
}

void SDQ_OptD::opt_Q_C(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                  float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]){      
    float divisor=0;
    float denominator=0;
    float val;
    int i,j;
    for(j=1; j<64; j++){  
        for(i=0; i<SDQ_OptD::seq_len_C; i++){
            divisor += seq_dct_coefs_Cb[i][j]*seq_dct_idxs_Cb[i][j]*SDQ_OptD::Block.Sen_Map[1][j];
            divisor += seq_dct_coefs_Cr[i][j]*seq_dct_idxs_Cr[i][j]*SDQ_OptD::Block.Sen_Map[2][j];
            denominator += pow(seq_dct_idxs_Cb[i][j],2)*SDQ_OptD::Block.Sen_Map[1][j];
            denominator += pow(seq_dct_idxs_Cr[i][j],2)*SDQ_OptD::Block.Sen_Map[2][j];
        }
        if(denominator != 0){
            val = divisor/denominator;
            val = round(MinMaxClip(val, MINQVALUE, MAXQVALUE));
            SDQ_OptD::Q_table_C[j] = val;
        }    
        else    Q_table_C[j] = MAXQVALUE;
        divisor=0;denominator=0; 
    }
}

void SDQ_OptD::opt_Q_Y_DC(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    float numerator=0;
    float denominator=0;
    float val;
    for(int i=0; i<seq_len_Y; i++){
        numerator += seq_dct_coefs_Y[i][0]*seq_dct_idxs_Y[i][0];
        denominator += pow(seq_dct_idxs_Y[i][0],2);
    }
    if(denominator != 0){
        val = round(numerator/denominator);
        val = MinMaxClip(val, MINQVALUE, MAXQVALUE);
        Q_table_Y[0] = val;
    }
    else    Q_table_Y[0] = MAXQVALUE;
}

void SDQ_OptD::opt_Q_C_DC(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
                     float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64]){      
    float numerator=0;
    float denominator=0;
    float val;
    for(int i=0; i<seq_len_C; i++){
        numerator += seq_dct_coefs_Cb[i][0]*seq_dct_idxs_Cb[i][0]*Block.Sen_Map[1][0];
        numerator += seq_dct_coefs_Cr[i][0]*seq_dct_idxs_Cr[i][0]*Block.Sen_Map[2][0];
        denominator += pow(seq_dct_idxs_Cb[i][0],2)*Block.Sen_Map[1][0];
        denominator += pow(seq_dct_idxs_Cr[i][0],2)*Block.Sen_Map[2][0];
    }
    if(denominator != 0){
        val = round(numerator/denominator);
        val = MinMaxClip(val, MINQVALUE, MAXQVALUE);
        Q_table_C[0] = val;
    }
    else    Q_table_C[0] = MAXQVALUE;
}

void SDQ_OptD::opt_DC_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    // calculate entropy rate for different size groups
    float ent[12];
    for(int s=0; s<12; s++){
        if(P_DC_Y[s]==0)    ent[s] = -log2(1/seq_len_Y)+s;
        else    ent[s] = -log2(P_DC_Y[s]/seq_len_Y)+s;
    }

    // DC trellis
    auto v = new float[seq_len_Y][33];
    auto cost_mini = new float[seq_len_Y][33];
    auto opt_pre = new int[seq_len_Y][33];     // store the j value of the optimal predecessor
    // set v values
    for(int i=0; i<seq_len_Y; i++){
        v[i][16] = round(seq_dct_coefs_Y[i][0]/Q_table_Y[0]);   // HDQ DC index
        for(int j=0; j<16; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
        for(int j=17; j<33; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
    }
    // initialize cost_mini[0][:]
    for(int j=0; j<33; j++){
        int s = size_group(v[0][j],11,0);
        cost_mini[0][j] = Beta_S*Block.Sen_Map[0][0]*pow(seq_dct_coefs_Y[0][0]-Q_table_Y[0]*v[0][j],2)+ent[s];
    }
    // figure out cost_mini and opt_pre for all states in all stages
    for(int i=1; i<seq_len_Y; i++){
        for(int j=0; j<33; j++){
            float comp_cost[33];
            for(int k=0; k<33; k++){    // k is j', which specifies the incoming path
                comp_cost[k] = cost_mini[i-1][k];
                int s = size_group(v[i][j]-v[i-1][k],11,0);   // calculate the size group of DIFF
                float cost_inc;
                cost_inc = Beta_S*Block.Sen_Map[0][0]*pow(seq_dct_coefs_Y[i][0]-Q_table_Y[0]*v[i][j],2)+ent[s];     // calculate cost_inc
                comp_cost[k] += cost_inc;   // add cost_inc to cost_mini
            }     
            cost_mini[i][j] = *min_element(comp_cost, comp_cost+33);
            opt_pre[i][j] = min_element(comp_cost, comp_cost+33)-comp_cost;
        }
    }

    // update DC indices
    int flag = min_element(cost_mini[seq_len_Y-1], cost_mini[seq_len_Y-1]+33)-cost_mini[seq_len_Y-1];
    seq_dct_idxs_Y[seq_len_Y-1][0] = v[seq_len_Y-1][flag];
    // backtrack
    for (int i=seq_len_Y-1; i > 0; i--) {
        seq_dct_idxs_Y[i-1][0] = v[i-1][opt_pre[i][flag]];
        flag = opt_pre[i][flag];
    }
    delete []v;
    delete []cost_mini;
    delete []opt_pre;
}

void SDQ_OptD::opt_DC_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64], 
                   float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    // calculate entropy rate for different size groups
    float ent[12];
    for(int s=0; s<12; s++){
        if(P_DC_C[s]==0)    ent[s] = -log2(1/seq_len_C)+s;
        else    ent[s] = -log2(P_DC_C[s]/seq_len_C)+s;
    }

    auto v = new float[seq_len_C][33];
    auto cost_mini = new float[seq_len_C][33];
    auto opt_pre = new int[seq_len_C][33];

    // Cb channel
    for(int i=0; i<seq_len_C; i++){
        v[i][16] = round(seq_dct_coefs_Cb[i][0]/Q_table_C[0]);
        for(int j=0; j<16; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
        for(int j=17; j<33; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
    }

    for(int j=0; j<33; j++){
        int s = size_group(v[0][j],11,0);
        cost_mini[0][j] = Beta_W*Block.Sen_Map[1][0]*pow(seq_dct_coefs_Cb[0][0]-Q_table_C[0]*v[0][j],2)+ent[s];
    }

    for(int i=1; i<seq_len_C; i++){
        for(int j=0; j<33; j++){
            float comp_cost[33];
            for(int k=0; k<33; k++){
                comp_cost[k] = cost_mini[i-1][k];
                int s = size_group(v[i][j]-v[i-1][k],11,0);
                float cost_inc;
                cost_inc = Beta_W*Block.Sen_Map[1][0]*pow(seq_dct_coefs_Cb[i][0]-Q_table_C[0]*v[i][j],2)+ent[s];
                comp_cost[k] += cost_inc;
            }     
            cost_mini[i][j] = *min_element(comp_cost, comp_cost+33);
            opt_pre[i][j] = min_element(comp_cost, comp_cost+33)-comp_cost;
        }
    }

    int flag = min_element(cost_mini[seq_len_C-1], cost_mini[seq_len_C-1]+33)-cost_mini[seq_len_C-1];
    seq_dct_idxs_Cb[seq_len_C-1][0] = v[seq_len_C-1][flag];
    for (int i=seq_len_C-1; i > 0; i--) {
        seq_dct_idxs_Cb[i-1][0] = v[i-1][opt_pre[i][flag]];
        flag = opt_pre[i][flag];
    }

    // Cr channel
    for(int i=0; i<seq_len_C; i++){
        v[i][16] = round(seq_dct_coefs_Cr[i][0]/Q_table_C[0]);
        for(int j=0; j<16; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
        for(int j=17; j<33; j++){v[i][j] = MinMaxClip(v[i][16]-16+j,-1023,1023);}
    }

    for(int j=0; j<33; j++){
        int s = size_group(v[0][j],11,0);
        cost_mini[0][j] = Beta_X*Block.Sen_Map[2][0]*pow(seq_dct_coefs_Cr[0][0]-Q_table_C[0]*v[0][j],2)+ent[s];
    }

    for(int i=1; i<seq_len_C; i++){
        for(int j=0; j<33; j++){
            float comp_cost[33];
            for(int k=0; k<33; k++){
                comp_cost[k] = cost_mini[i-1][k];
                int s = size_group(v[i][j]-v[i-1][k],11,0);
                float cost_inc;
                cost_inc = Beta_X*Block.Sen_Map[2][0]*pow(seq_dct_coefs_Cr[i][0]-Q_table_C[0]*v[i][j],2)+ent[s];
                comp_cost[k] += cost_inc;
            }     
            cost_mini[i][j] = *min_element(comp_cost, comp_cost+33);
            opt_pre[i][j] = min_element(comp_cost, comp_cost+33)-comp_cost;
        }
    }

    flag = min_element(cost_mini[seq_len_C-1], cost_mini[seq_len_C-1]+33)-cost_mini[seq_len_C-1];
    seq_dct_idxs_Cr[seq_len_C-1][0] = v[seq_len_C-1][flag];

    for (int i=seq_len_C-1; i > 0; i--) {
        seq_dct_idxs_Cr[i-1][0] = v[i-1][opt_pre[i][flag]];
        flag = opt_pre[i][flag];
    }
    delete []v;
    delete []cost_mini;
    delete []opt_pre;
}