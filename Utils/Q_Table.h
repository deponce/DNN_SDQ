// Q_table.h

// MIT License

// Copyright (c) 2022 Ahmed Hussein Salamah, deponce(Linfeng Ye), Kaixiang Zheng, University of Waterloo

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

   // for(i = 0; i < 5; ++i)
   // variance += pow(val[i] - mean, 2);
   // variance=variance/5;
   // stdDeviation = sqrt(variance);


void parameterCal(float varianceData[64], float lambdaData[64],float seq_dct_coefs[][64], int N_block)
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

// int binarySearch(float arr, int l, int r, int x)
// {
//     if (r >= l) {
//         int mid = l + (r - l) / 2;
  
//         // If the element is present at the middle
//         // itself
//         if (arr[mid] == x)
//             return mid;
  
//         // If element is smaller than mid, then
//         // it can only be present in left subarray
//         if (arr[mid] > x)
//             return binarySearch(arr, l, mid - 1, x);
  
//         // Else the element can only be present
//         // in right subarray
//         return binarySearch(arr, mid + 1, r, x);
//     }
  
//     // We reach here when element is not
//     // present in array
//     return -1;
// }

void quantizationTable_OptD(float seq_dct_coefs[][64], float Q_Table[64], int N_block, float d_waterLevel, int QMAX_Y)
{
    // const int QMAX_Y = 46;
    // float d_waterLevel;
    // d_waterLevel = 18.5;

    float varianceData[64]; // No need for DC 
    float lambdaData[64];
    auto Dlap = new float[QMAX_Y + 1][64];
    parameterCal(varianceData, lambdaData , seq_dct_coefs ,N_block);
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
        // Q_Table[i] = binarySearch(Dlap[i], 0, N_block - 1, d_waterLevel);
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