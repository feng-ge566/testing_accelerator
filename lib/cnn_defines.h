#include <stdint.h>

#ifndef _CNN_DEFINES_
#define _CNN_DEFINES_

#define Tin_int8 16
#define log2_Tin_int8 4
#define base_Tin 2*Tin_int8
#define base_log2Tin log2_Tin_int8
//#define Tout 16   //NOTE: Tout<=Tin_int8  //Parallel factor of CH_out,
#define log2Tout 4		
#define Tout 16
#define log2_other 6
#define log2_scale 6
#define log2_KyKx 6

#define MAX_DW 8
#define MAX_DW2 2*MAX_DW

//two RAM blocks, one for wt, the other one for dat, both are with 1024 depth
#define WT_SRAM_DEPTH 1152*2
#define log2WT_SRAM_DEPTH 11+1
#define WT_SRAM_WIDTH  base_Tin*MAX_DW

#define OUTPUT_SRAM_DEPTH 128*2
#define log2OUTPUT_SRAM_DEPTH 7+1
#define OUTPUT_SRAM_WIDTH Tout*MAX_DW

#define FMC_BUS_IN_WIDTH 16
#define log2FMC_BUS_IN_WIDTH 4
#define FMC_BUS_OUT_WIDTH 8
#define log2FMC_BUS_OUT_WID 3
#define SRAM_FMC_WIDTH WT_SRAM_WIDTH/FMC_BUS_IN_WIDTH

#endif
