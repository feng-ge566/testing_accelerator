#include <stdint.h>
//structure of VGG16:           0,   1,     2,    3,
uint32_t conv_Hin[]        = { 1, 16,  16,    16,   16};
uint32_t conv_Win[]        = { 1, 16,  16,    16,   16};
uint32_t conv_CHin[]       = { 16, 16,  64,   128,  256};
uint32_t conv_CHout[]      = { 16, 16,  16,    16,   16};
uint32_t conv_Ky[]         = {  3,  3,   3,     3,    3};
uint32_t conv_Kx[]         = {  3,  3,   3,     3,    3};
uint32_t Sx[]              = {  1,  1,   1,     1,    1};
uint32_t Sy[]              = {  1,  1,   1,     1,    1};
uint32_t pad_left[]        = {  1,  1,   1,     1,    1};
uint32_t pad_right[]       = {  1,  1,   1,     1,    1};
uint32_t pad_up[]          = {  1,  1,   1,     1,    1};
uint32_t pad_down[]        = {  1,  1,   1,     1,    1};
                                                                          
//structure of VGG16:          0,  0,   1,       2,   3, 
uint32_t conv_wt_scale[]   = { 0,  0,   0,       0,   0};
uint32_t conv_in_scale[]   = { 0,  0,   0,       0,   0};
uint32_t conv_out_scale[]  = { 0,  0,   0,       0,   0};
uint32_t conv_wt_bit[]     = { 8,  8,   4,       2,   1};
uint32_t conv_in_bit[]     = { 8,  8,   4,       2,   1};
uint32_t conv_out_bit[]    = { 8,  8,   8,       8,   8};
uint32_t conv_relu_en[]    = { 0,  0,   0,       0,   0};


