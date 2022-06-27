// #include<map>
// #include<math.h>
// #include<algorithm>
#include "SDQ_Class.h"
// #include "../Utils/utils.h"
// #include "../EntCoding/EntUtils.h"

void SDQ::opt_Q_Y_DC(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
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
}

void SDQ::opt_Q_C_DC(float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64],
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
}

void SDQ::opt_DC_Y(float seq_dct_idxs_Y[][64], float seq_dct_coefs_Y[][64]){
    // calculate entropy rate for different size groups
    float ent[12];
    for(int s=0; s<12; s++){
        ent[s] = min(-log2(P_DC_Y[s]/P_DC_Y[TOTAL_KEY])+s,float(1e10));
    }

    // DC trellis
    float v[seq_len_Y][33];
    float cost_mini[seq_len_Y][33];
    int opt_pre[seq_len_Y][33];     // store the j value of the optimal predecessor
    // set v values
    for(int i=0; i<seq_len_Y; i++){
        v[i][16] = round(seq_dct_coefs_Y[i][0]/Q_table_Y[0]);   // HDQ DC index
        for(int j=0; j<16; j++){v[i][j] = v[i][16]-16+j;}
        for(int j=17; j<33; j++){v[i][j] = v[i][16]-16+j;}
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
}

void SDQ::opt_DC_C(float seq_dct_idxs_Cr[][64], float seq_dct_coefs_Cr[][64], 
                   float seq_dct_idxs_Cb[][64], float seq_dct_coefs_Cb[][64]){
    // calculate entropy rate for different size groups
    float ent[12];
    for(int s=0; s<12; s++){
        ent[s] = min(-log2(P_DC_C[s]/P_DC_C[TOTAL_KEY])+s,float(1e10));
    }

    float v[seq_len_C][33];
    float cost_mini[seq_len_C][33];
    int opt_pre[seq_len_C][33];

    // Cb channel
    for(int i=0; i<seq_len_C; i++){
        v[i][16] = round(seq_dct_coefs_Cb[i][0]/Q_table_C[0]);
        for(int j=0; j<16; j++){v[i][j] = v[i][16]-16+j;}
        for(int j=17; j<33; j++){v[i][j] = v[i][16]-16+j;}
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
        for(int j=0; j<16; j++){v[i][j] = v[i][16]-16+j;}
        for(int j=17; j<33; j++){v[i][j] = v[i][16]-16+j;}
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
}