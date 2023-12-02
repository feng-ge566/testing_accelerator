#include <stdint.h>
uint32_t batch_size=16;
//structure of VGG16:           0,   1,       2,   3,       4,     5,   6,          7,   8,     9,          10,  11,  12,           13,  14,   15
uint32_t conv_Hin[]        = { 224, 224,    112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           1,    1,    1 };   //16,
uint32_t conv_Win[]        = { 224, 224,    112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,          16,   16,   16 };   //32,
uint32_t conv_CHin[]       = {   3,  64,      64, 128,     128,   256, 256,        256, 512,   512,        512, 512, 512,     512*7*7, 4096,4096 };  //64,
uint32_t conv_CHout[]      = {  64,  64,     128, 128,     256,   256, 256,        512, 512,   512,        512, 512, 512,        4096, 4096,1000 };  //64,
uint32_t conv_Ky[]         = {   3,   3,       3,   3,       3,     3,   3,          3,   3,     3,          3,   3,   3,           1,    1,   1 };  // 3,
uint32_t conv_Kx[]         = {   3,   3,       3,   3,       3,     3,   3,          3,   3,     3,          3,   3,   3,           1,    1,   1 };  // 3,
uint32_t Sx[]              = {   1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           1,    1,   1 };  // 1,
uint32_t Sy[]              = {   1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           1,    1,   1 };  // 1,
uint32_t pad_left[]        = {   1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };  // 1,
uint32_t pad_right[]       = {   1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };  // 1,
uint32_t pad_up[]          = {   1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };  // 1,
uint32_t pad_down[]        = {   1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };  // 1,
uint32_t conv_Hout[]       = { 224, 224,     112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           1,    1,   1 };  //16,
uint32_t conv_Wout[]       = { 224, 224,     112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           1,    1,   1 };  //32,


//structure of VGG16:          0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15
// uint32_t conv_in_scale[]   = { 3,    3,   3,   3,   3,   3,   3,   3,   0,   3,   3,   3,   3,   3,   3,   3 }; //

// uint32_t conv_wt_scale[]   = { 6,    6,   6,   5,   7,   5,   7,   7,   0,   7,   7,   6,   7,   9,   8,   9 }; //
uint32_t conv_in_scale[]   = {0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   0,   4}; //
uint32_t conv_wt_scale[]   = {0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   0,   4}; //

uint32_t conv_bias_scale[] = { 5,    6,   7,   6,   7,   7,   7,   8,   0,   8,   9,   9,   8,  10,   8,   9 }; //

uint32_t conv_out_scale[]  = { 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }; //

uint32_t conv_wt_bit[]     = { 8,    4,   4,   2,   4,   2,   4,   4,   2,   4,   4,   2,   4,   4,   4,   8 }; //
uint32_t conv_in_bit[]     = { 8,    4,   4,   2,   4,   2,   4,   4,   2,   4,   4,   2,   4,   4,   4,   8 }; //
uint32_t conv_out_bit[]    = { 8,    8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8 }; //

uint32_t conv_relu_en[]    = { 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 }; //

uint32_t pool_in_bit[]     = {          4,              4,                       8,                     2,                   2 };
uint32_t pool_out_bit[]    = {          4,              4,                       8,                     2,                   2 };
uint32_t pool_in_scale[]   = {          2,              1,                      -1,                    -4,                  -4 };
uint32_t pool_out_scale[]  = {          2,              1,                      -1,                    -4,                  -4 };

uint32_t pool_Hin[]        = {          224,           112,                     56,                    28,                  14 };
uint32_t pool_Win[]        = {          224,           112,                     56,                    28,                  14 };
uint32_t pool_CHin[]       = {          64,            128,                     256,                   512,                 512 };
uint32_t pool_CHout[]      = {          64,            128,                     256,                   512,                 512 };
uint32_t pool_Ky[]         = {          2,              2,                       2,                     2,                   2 };
uint32_t pool_Kx[]         = {          2,              2,                       2,                     2,                   2 };
uint32_t pool_Sx[]         = {          2,              2,                       2,                     2,                   2 };
uint32_t pool_Sy[]         = {          2,              2,                       2,                     2,                   2 };
uint32_t pool_pad_left[]   = {          0,              0,                       0,                     0,                   0 };
uint32_t pool_pad_right[]  = {          0,              0,                       0,                     0,                   0 };
uint32_t pool_pad_up[]     = {          0,              0,                       0,                     0,                   0 };
uint32_t pool_pad_down[]   = {          0,              0,                       0,                     0,                   0 };
uint32_t pool_Hout[]       = {          112,            56,                      28,                    14,                  7 };
uint32_t pool_Wout[]       = {          112,            56,                      28,                    14,                  7 };

char *weight_filelist[] = {
"./test_model_bin_#2/conv_layer00_8bit_weight_s=-6.0.bin",
"./test_model_bin_#2/conv_layer01_4bit_weight_s=-6.0.bin",
"./test_model_bin_#2/conv_layer03_4bit_weight_s=-6.0.bin",
"./test_model_bin_#2/conv_layer04_2bit_weight_s=-5.0.bin",
"./test_model_bin_#2/conv_layer06_4bit_weight_s=-7.0.bin",
"./test_model_bin_#2/conv_layer07_2bit_weight_s=-6.0.bin",
"./test_model_bin_#2/conv_layer08_4bit_weight_s=-7.0.bin",
"./test_model_bin_#2/conv_layer10_4bit_weight_s=-7.0.bin",
"./test_model_bin_#2/conv_layer11_2bit_weight_s=-6.0.bin",
"./test_model_bin_#2/conv_layer12_4bit_weight_s=-8.0.bin",
"./test_model_bin_#2/conv_layer14_4bit_weight_s=-8.0.bin",
"./test_model_bin_#2/conv_layer15_2bit_weight_s=-6.0.bin",
"./test_model_bin_#2/conv_layer16_4bit_weight_s=-8.0.bin",
"./test_model_bin_#2/fc_layer00_4bit_weight_s=-10.0.bin",
"./test_model_bin_#2/fc_layer01_4bit_weight_s=-8.0.bin",
"./test_model_bin_#2/fc_layer02_8bit_weight_s=-9.0.bin"
 };

char *feature_filelist[] = {
"./test_model_bin_#2/conv_layer00_8bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer01_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer03_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer04_2bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer06_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer07_2bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer08_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer10_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer11_2bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer12_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer14_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer15_2bit_act_s=-3.0.bin",
"./test_model_bin_#2/conv_layer16_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/fc_layer00_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/fc_layer01_4bit_act_s=-3.0.bin",
"./test_model_bin_#2/fc_layer02_8bit_act_s=-3.0.bin"
 };
