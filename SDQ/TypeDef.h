// TypeDef.h

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

#ifndef TYPEDEF_H
#define TYPEDEF_H


#define PRINT_FIND_SEGMENTS_DECODER      0
#define PRINT_HUFFMAN_TABLE              0
#define PRINT_QUANTIZATION_TABLE         0
#define PRINT_FRAME_HEADER_SOF           0
#define PRINT_SOS                        0
#define PRINT_BLOCK_PROGRESS             0
#define DEBUGLEVEL                       12

#define OPEN_CV_ENABLED                  0
#define DISPLAY_BMP_IMAGE                0

// Component indices
#define COMPONENT_Y                      0
#define COMPONENT_Cb                     1
#define COMPONENT_Cr                     2

// Indices for the default huffman table
#define Y_DC_IDX                            0
#define CbCr_DC_IDX                         1
#define Y_AC_IDX                            2
#define CbCr_AC_IDX                         3

// Maximum number of supported color components
#define MAX_NUMBER_SUPPORTED_COLOR_COMPONENTS     3

// Table length for writing two quantization tables back to back with two DQT markers
#define DQT_LENGTH_WITH_TWO_MARKERS      67


// Table length for writing two quantization tables back to back with one DQT marker
#define DQT_LENGTH_WITH_ONE_MARKERS      132

// JPEG Buffer maximum output size to write sections of the bitstream instead of byte by byte
#define JPEG_OUT_HEADER_SIZE			 500

// Is Fast way of writing in the encoder?
#define IS_JPEG_ENCODER_WRITE_FAST       1

// Threshold that flag of the count block outlier to to be non-black block
#define NON_BLACK						 2

#define QFACTOR                           10
#define IS_DEFAULT_QTABLE                 1
#define IS_ONLY_TCM                       1
#define TCM_OUTLIER_THRESHOLD             0


// profile encoder?
#define PROFILE_JPEG_SAVE_PIC               0


// profile decoder?
#define PROFILE_JPEG_DECODE_PIC             0


// When you enable customized huffman table, you should set the following TWO flags to 1 (next flag MUST ALWAYS BE 1)
#define IS_ENABLE_USE_DEFAULT_HUFF_TABLES    1

// use default huffman table or customized?
#define CUSTOMIZED_HUFFMAN_TABLE   1

// use Qtable from the input picture in the encoder?
#define IS_ENABLE_USE_QTABLE_FROM_PICTURE    0

// how many bytes to skip while QFACTOR Experiment
#define SKIP_BYTES_Q_FACTOR_EXP_RGB					    (0x44 + 0x43)
#define SKIP_BYTES_Q_FACTOR_EXP_BLKANDWHITE             (0x42)



// Defines a tuple of length and code, for use in the Huffman maps
typedef std::pair<int, unsigned short> huffKey;

#if IS_ENABLE_CUSTOMIZED_HUFF_TABLES

unsigned char code_length_freq[4][16];

#else

//static unsigned char code_length_freq[4][16];

// Progressive Encoder - Default Huffman Tables
static unsigned char code_length_freq[4][16] = {
    { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 },
    { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125 },
    { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119 }
};
#endif

// Define some types for huffman encoding
typedef unsigned char  uint8;
typedef signed short   int16;
typedef signed int     int32;
typedef unsigned short uint16;
typedef unsigned int   uint32;
typedef unsigned int   uint;


static const unsigned char kDCSyms[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
static const unsigned char kACSyms[2][162] = {
    { 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa },
    { 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa }
};
#endif /* TypeDef_h */