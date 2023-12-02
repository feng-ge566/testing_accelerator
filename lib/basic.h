#include "cnn_defines.h"
#include <stdbool.h>
#include <stdint.h>

#define XPAR_DMA_0_BASEADDR 0xA0000000
#define RESET_VSC   1
#define START_VSC   2
#define CFG_REG     3
#define WRITE_DAT   4
#define WRITE_WET   5
#define MODE_SEL    6
#define TEST_REG    8
#define CLK_SEL     11
#define CLK_DVI_NUM 12
#define AXI_MODEL   13

#define READ_ADDR     7
#define READ_ODAT     7
#define OUT_CNT       8
#define GET_DONE      9
#define GET_OUT_VALID 11
#define GET_CFG_REG   12

#define GET_STATE     10

typedef struct conv_cfg {
	uint32_t CHin, CHout;
	uint32_t Hin, Hout;
	uint32_t Win, Wout;
	uint32_t Ky, Kx;
	uint32_t Sy, Sx;
	uint32_t Py, Px;
	uint32_t DAT_DW_L0;
	uint32_t DAT_DW_L1;

	uint32_t OUT_Shift;
	uint32_t RELU_EN_L0;

	uint32_t pixel_group_num_minus1;

	uint32_t WT_DW_L0;
	uint32_t Tin_L0;

    uint32_t slice_of_Wout_x_Hout;
    uint32_t total_output_pixels;

    uint32_t slice_of_CHout_L0;
    uint32_t slice_of_CHin_L0;
    uint32_t slice_of_Tin_div_Tout_L0;


    uint8_t div_num;
    uint8_t Tin_factor;
    uint8_t out_data_width;
    uint32_t acc_times_minus1;


    uint32_t input_data_num;
    uint32_t output_data_num;
    uint32_t practical_log2_cycle;

    uint32_t data_shape;
    uint32_t dw_num, dw, Tin_l0, Tin_num; //using pingjie
    
    uint8_t freqset, divset, divset_pad;

} conv_cfg, * conv_cfg_p;

void FPGA_Init();
void Xil_Out32(int addr, uint32_t data);
uint32_t Xil_In32(int addr);


void ShowConfig(conv_cfg *cfg);
void InitConfig(conv_cfg *cfg, 
		uint32_t CHin, uint32_t Hin, uint32_t Win, uint32_t CHout, uint32_t Ky, uint32_t Kx,
		uint32_t Py  , uint32_t Px , uint32_t Sy , uint32_t Sx   , uint32_t DAT_DW_L0,
		uint32_t DAT_DW_L1, uint32_t  RELU_EN_L0, uint32_t OUT_Shift,uint32_t practical_log2_cycle,
        uint8_t freqset, uint8_t divset, uint8_t divset_pad);

void conv_soft(conv_cfg *cfg , int8_t *addr_feature,  int8_t *addr_weight, int8_t *CONV_SOLFT);
void conv_div(conv_cfg *cfg, int8_t *addr_feature, int8_t *addr_weight,
                                    uint32_t *BASE_DAT, int32_t len_re_feature,
                                    uint32_t *BASE_WT, int32_t len_re_weight);
int in_index(struct conv_cfg *cfg, uint32_t i, uint32_t j, uint32_t k);

int wt_index(struct conv_cfg *cfg, uint32_t i, uint32_t j, uint32_t k, uint32_t l);

int out_index(struct conv_cfg *cfg, uint32_t i, uint32_t j, uint32_t k);

void Rand_Gen(conv_cfg *cfg ,int8_t *feature_rand, int len_rand_feature, int8_t *weight_rand, int len_rand_weight,bool low_value_tests);
//void Read_data(conv_cfg *cfg , char *feature_file, int8_t *read_feature, int len_read_feature, char *weight_file, int8_t *read_weight, int len_read_weight);
void Read_data(conv_cfg *cfg , char *feature_file, int8_t *read_feature, int len_read_feature, char *weight_file, int8_t *read_weight, int len_read_weight, uint32_t in_scale, uint32_t wt_scale);
void Read_bin_data(conv_cfg *cfg , char *feature_file, int8_t *read_feature, int len_read_feature, char *weight_file, int8_t *read_weight, int len_read_weight, uint32_t in_scale, uint32_t wt_scale);
void config_on_chip_clock(conv_cfg *cfg, uint32_t CFG_REG_ADDR);

void config_on_chip_register(conv_cfg *cfg, uint32_t CFG_REG_ADDR, uint32_t *cmd_cfg);
void check_on_chip_register(conv_cfg *cfg, uint32_t CFG_REG_ADDR, uint32_t GET_CFG_REG_ADDR,uint32_t MODE_SEL_ADDR, uint8_t *addr_reg);
uint8_t read_on_chip_register(uint32_t CFG_REG_ADDR, uint32_t GET_CFG_REG_ADDR, uint32_t MODE_SEL_ADDR, uint8_t addr_reg);
void read_all_on_chip_register(uint32_t CFG_REG_ADDR, uint32_t GET_CFG_REG_ADDR, uint32_t MODE_SEL_ADDR);

void start_vsc(uint32_t START_VSC_ADDR);
bool waiting_finish(uint32_t GET_OUT_VALID_ADDR, uint32_t GET_DONE_ADDR);
void Write_feature(conv_cfg *cfg, uint32_t *feature, uint32_t write_addr);
void Write_weight(conv_cfg *cfg, uint32_t *weight, uint32_t write_addr);

void read_chip_out_result(conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR);
void read_chip_out_result_div(conv_cfg *cfg_div, conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR, uint32_t test_i, uint32_t test_j);
void read_chip_out_result_div_faster(conv_cfg *cfg_div, conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR, uint32_t test_i, uint32_t test_j, uint32_t div_KxKyCin);
void read_chip_out_result_div_linera(conv_cfg *cfg_div, conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR, 
                                     uint32_t test_i, uint32_t test_j, uint32_t div_KxKyCin, uint32_t test_k);
void compare_result(conv_cfg *cfg, char *d0_name, int8_t *chip_out_result, char *d1_name, int8_t *conv_solft, bool out_error,bool print_all);


void RunConv_Reshape(conv_cfg *cfg, int8_t *addr_feature, int8_t *addr_weight,
                                    uint32_t *BASE_DAT, int32_t len_re_feature,
                                    uint32_t *BASE_WT, int32_t len_re_weight );
void Reshape_conv_soft(conv_cfg *cfg, int8_t *addr_feature,  int8_t *addr_weight, int8_t *addr_re_conv);
void test_malloc();
