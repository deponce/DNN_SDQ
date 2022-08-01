// Q_table.h

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

#include <math.h>
#include <algorithm>
using namespace std;


void minMaxQuantizationStep(int colorspace, float &MINQVALUE, float &MAXQVALUE, float &QUANTIZATION_SCALE)
{
    if(colorspace == 0) // HDQ
    { 
        // YUV
        MINQVALUE = 1.;
        MAXQVALUE = 255.;
        QUANTIZATION_SCALE = 1.;   
    }
    else if(colorspace == -1) // SDQ
    { 
        // YUV
        MINQVALUE = 5.;
        MAXQVALUE = 255.;
        QUANTIZATION_SCALE = 1.;   
    }
    else if(colorspace == 1) // SWX
    {
        // Setting 1
        QUANTIZATION_SCALE = 3.;
        MAXQVALUE = 255.*sqrt(QUANTIZATION_SCALE);
        MINQVALUE = 1.;

    }
    else if(colorspace == 2)  // SWX
    {
        // Setting 2
        MINQVALUE = 1.; 
        MAXQVALUE = 255.;
        QUANTIZATION_SCALE = 3.;

    }
    else if(colorspace == 3)  // SWX remove the mean form each image
    {

        // Setting 1
        // QUANTIZATION_SCALE = 3.;
        // MAXQVALUE = 255.*sqrt(QUANTIZATION_SCALE);
        // MINQVALUE = 1.;

        // Setting 2
        // MAXQVALUE = 422.;
        // QUANTIZATION_SCALE = 3.;
        // MINQVALUE = sqrt(QUANTIZATION_SCALE); 

        // Setting 3
        QUANTIZATION_SCALE = 3.;
        MAXQVALUE = 255.*sqrt(QUANTIZATION_SCALE);
        MINQVALUE = 1./sqrt(QUANTIZATION_SCALE);;

    }
    else 
    {
        MINQVALUE = 1.; 
        MAXQVALUE = 255.;
        QUANTIZATION_SCALE = 1.;

    }
}



void parameterCal_Y(float varianceData[64], float lambdaData[64],float seq_dct_coefs[][64], int N_block)
{
    float tmp;
    for (int i = 0; i < 64; i++ )
    {
        float variance = 0, mean = 0, lambda = 0;

        for (int j=0;j<N_block; j++)
        {
            mean += seq_dct_coefs[j][i];
        }
        mean /= N_block;

        for (int j=0;j<N_block; j++)
        {
            variance += pow(seq_dct_coefs[j][i] - mean, 2);
        }
        variance /= N_block;
        
        for (int j=0;j<N_block; j++)
        {
            tmp = seq_dct_coefs[j][i] - mean;
            lambda += abs(tmp);
        }
        varianceData[i] = variance;
        lambdaData[i] = lambda/N_block;

        // cout << varianceData[i] << "\t" << lambdaData[i] << "\n";
    }
    // cout << endl;
}

void parameterCal_C(float varianceData[64], float lambdaData[64],float seq_dct_coefs_Cb[][64], float seq_dct_coefs_Cr[][64], int N_block)
{
    float tmp;
    for (int i = 0; i < 64; i++ )
    {
        float variance = 0, mean = 0, lambda = 0;

        // Cal the Mean for both Cb Cr
        for (int j=0;j<N_block; j++)
        {
            mean += seq_dct_coefs_Cb[j][i];
        }
        for (int j=0;j<N_block; j++)
        {
            mean += seq_dct_coefs_Cr[j][i];
        }
        mean /= (2*N_block);

        // Cal the variance for both Cb Cr
        for (int j=0;j<N_block; j++)
        {
            variance += pow(seq_dct_coefs_Cb[j][i] - mean, 2);
        }
        for (int j=0;j<N_block; j++)
        {
            variance += pow(seq_dct_coefs_Cr[j][i] - mean, 2);
        }
        variance /= (2*N_block);
        
        // Cal the lambda for both Cb Cr
        for (int j=0;j<N_block; j++)
        {
            tmp = seq_dct_coefs_Cb[j][i] - mean;
            lambda += abs(tmp);
        }
        for (int j=0;j<N_block; j++)
        {
            tmp = seq_dct_coefs_Cr[j][i] - mean;
            lambda += abs(tmp);
        }

        varianceData[i] = variance;
        lambdaData[i] = lambda/ (2*N_block);

        // cout << varianceData[i] << "\t" << lambdaData[i] << "\n";
    }
    // cout << endl;
}

float Cal_DT(float d_waterLevel, float varianceData[64])
{
    float DT = 0;
    for (int i = 0; i <= 64; i++)
    {
        if(d_waterLevel <= varianceData[i]) DT += d_waterLevel;
        else DT += varianceData[i];
    }
    return DT;
}


float Cal_d(float DT, float varianceData[64])
{
    float sum_var = 0;
    for (int i = 0; i <=64; i++)
    {
        sum_var += varianceData[i];
    }
    DT = MinMaxClip(DT, 0, sum_var); // clip between 0 and sum_var

    // bi-section search
    float eps = 1e-5;
    float a = 0;
    // Maximum variance allover 64 Cofficient
    float b = *max_element(varianceData, varianceData + 64);

    if (DT == 0) return a;
    // I believe it should be DT>= sum_var
    if (DT == sum_var) return b;

    float c = a;
    while ((b-a) >= eps) {
        // Find middle point
        c = (a+b)/2;
        // Check if middle point is root
        if (Cal_DT(c,varianceData) == DT)
            break;
        // Decide the side to repeat the steps
        else if ((Cal_DT(c,varianceData)-DT)*(Cal_DT(a,varianceData)-DT) < 0)
            b = c;
        else
            a = c;
    }
    return c;
}

void OptD(float varianceData[64], float lambdaData[64], float Q_Table[64], float DT, float& d_waterLevel, int QMAX_Y)
{
    auto Dlap = new float[QMAX_Y + 1][64];

    if (d_waterLevel < 0) d_waterLevel = Cal_d(DT, varianceData); // DT will be used only if d_waterLevel < 0
    

    float si, q_lambda;
    float p1, p2, p3;
    // cout <<  "Q value" << "\t"  << "Variance" << "\t" << "lambda" << "\n";
    lambdaData[0] = 0.0;
    for (int i = 1; i < 64; i++)
    {
        Dlap[0][i] = 0.0;
        for (int q = 1; q <= QMAX_Y; q++)
        {
            q_lambda = q / lambdaData[i];
            si = (q - lambdaData[i]) + ( q / (exp(q_lambda) - 1) );
            p1 = 2 * pow(lambdaData[i], 2);
            p2 = 2 * q * (lambdaData[i] + si -0.5 * q);
            p3 = exp(si/lambdaData[i]) * (1 - exp(-1 * q_lambda));
            Dlap[q][i] = p1 - (p2/p3);
            // cout << Dlap[q][i]  << "\n";
            // break;
        }
    }

    for (int i = 0; i < 64; i++)
    {
        if(varianceData[i] < d_waterLevel)
        {
            // Q_Table[i] = 255; // ACT as FAST QUANTIZTION
            Q_Table[i] = QMAX_Y;
        }
        else
        {
            if (i == 0) // DC q step
            {
                Q_Table[i] = min(floor(sqrt(12*d_waterLevel)), float(QMAX_Y));
            }
            else
            {

               for (int q = QMAX_Y; q >= 1 ; q--)
               {
                    // cout << Dlap[q][i]  << "\t" << d_waterLevel;
                    if (Dlap[q][i]  <= d_waterLevel)
                    {
                        Q_Table[i] = q;
                        break;
                    }
               }
            }      
        }
        // cout <<  Q_Table[i]  << "\t"  << varianceData[i] << "\t" << lambdaData[i] << "\n";
    }
}

void quantizationTable_OptD_C(float seq_dct_coefs_Cb[][64], float seq_dct_coefs_Cr[][64], float Q_Table[64], int N_block, float DT, float& d_waterLevel, int QMAX_Y)
{
    float varianceData[64];
    float lambdaData[64];   // No need for DC
    parameterCal_C(varianceData, lambdaData , seq_dct_coefs_Cb, seq_dct_coefs_Cr ,N_block);
    OptD(varianceData,lambdaData, Q_Table, DT, d_waterLevel,QMAX_Y);
    

}
void quantizationTable_OptD_Y(float seq_dct_coefs[][64], float Q_Table[64], int N_block, float DT, float& d_waterLevel, int QMAX_Y)
{
    float varianceData[64];
    float lambdaData[64];   // No need for DC
    parameterCal_Y(varianceData, lambdaData , seq_dct_coefs ,N_block);
    OptD(varianceData,lambdaData, Q_Table, DT, d_waterLevel,QMAX_Y);

}


void quantizationTable(int colorspace, float MINQVALUE,float MAXQVALUE, float QUANTIZATION_SCALE, int QF, bool Luminance, float Q_Table[64])
{
    QF = max(min(QF, 100),0);
    if(QF==0){
        QF=1;
    }
    float quantizationTableData_Y[64]={16.,  11.,  12.,  14.,  12.,  10.,  16.,  14.,  13.,  14.,  18.,  17.,
                                        16.,  19.,  24.,  40.,  26.,  24.,  22.,  22.,  24.,  49.,  35.,  37.,
                                        29.,  40.,  58.,  51.,  61.,  60.,  57.,  51.,  56.,  55.,  64.,  72.,
                                        92.,  78.,  64.,  68.,  87.,  69.,  55.,  56.,  80., 109.,  81.,  87.,
                                        95.,  98., 103., 104., 103.,  62.,  77., 113., 121., 112., 100., 120.,
                                        92., 101., 103.,  99.};
    float quantizationTableData_C[64]={17., 18., 18., 24., 21., 24., 47., 26., 26., 47., 99., 66., 56., 66.,
                                        99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.,
                                        99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.,
                                        99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99., 99.,
                                        99., 99., 99., 99., 99., 99., 99., 99.};
    float S;
    float q;              
    if(QF<50){
        S = 5000/QF;
    }
    else{
        S = 200-2*QF;
    }
    if (Luminance == true){
        for(int i=0; i<64; i++){
            q = (50.+S*quantizationTableData_Y[i])/100.;
            if ((colorspace == 3) ||( colorspace == 1)) // No Round
            {
                Q_Table[i] = MinMaxClip((q * sqrt(QUANTIZATION_SCALE)), MINQVALUE, MAXQVALUE);
            }
            else // JPEG Standard
            {
                Q_Table[i] = MinMaxClip(round(q * sqrt(QUANTIZATION_SCALE)), MINQVALUE, MAXQVALUE);
            }

        
        }
    }
    else{
        for(int i=0; i<64; i++){
            q = (50.+S*quantizationTableData_C[i])/100.;
            
            if ((colorspace == 3) ||( colorspace == 1)) // No Round
            {
                Q_Table[i] = MinMaxClip((q* sqrt(QUANTIZATION_SCALE)), MINQVALUE, MAXQVALUE);
            }
            else // JPEG
            {
                Q_Table[i] = MinMaxClip(round(q* sqrt(QUANTIZATION_SCALE)), MINQVALUE, MAXQVALUE);
            }
        }
    }
}