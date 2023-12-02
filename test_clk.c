#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "math.h"
#include <time.h>
#include <stdbool.h>
//#include "ff.h"

#include "cnn_defines.h"
#include "basic.h"
//#include "reshape.h"
//#include "func.h"
//#include "model_structure.h"
//#include "typical_structure.h
#include "rand_structure.h"

#include <unistd.h>
#include <getopt.h>
//#define EN_CACHE



int main(int argc, char* argv[])
{
	int test_case = 0;
    uint8_t freqset= 31;
    uint8_t divset = 7;
    uint8_t divset_pad =7;
	uint8_t cycle_num=1;
    bool ouput_error=false;
    bool print_ouput=false;
	bool low_value_tests=false;
    int ret = -1;
    int option_index = 0;
    struct option long_options[] =
    {  
        {"help"     , no_argument      , NULL, 'h'},
        {"testcase" , required_argument, NULL, 't'},
		{"freqset"  , required_argument, NULL, 'f'},
		{"divset"   , required_argument, NULL, 'd'},
		{"divsetpad", required_argument, NULL, 'b'},
		{"cycle"    , required_argument, NULL, 'c'},
		{"ouputerror"    , required_argument, NULL, 'o'},
		{"printouput"    , required_argument, NULL, 'p'},
		{"low_value_tests"    , required_argument, NULL, 'l'},
        {NULL,     0,                 NULL,  0}
    }; 
    while ((ret = getopt_long(argc, argv, "", long_options, &option_index)) != -1)
    {  
       switch (ret) {
        case 'h':
	        printf("--testcase=${testcase}, the test num in xx_strucutre.h, default = %d \n example --testcase=0\n \n",test_case);
			printf("--freqset=${freqset}, frequency set from 0~31, default = %d \nexample: --freqset=16 \n \n", freqset);
			printf("--divset=${divset}, frequency div set from 0~7, default = %d \nexample: --divset=5  \n \n", divset);
			printf("--divsetpad=${divsetpad}, frequency div set from 0~7, default = %d \n example: --divsetpad=5  \n \n", divset_pad);
			printf("--cycle=${cycle}, cycle num set from 0~31, default = %d \nexample: --cycle=10, set cycle num=2^10 \n \n",cycle_num);
			printf("--ouputerror=true/false, ouput error result default = %d \n \n", ouput_error);
			printf("--printouput=true/false, output all result, default = %d \n \n", print_ouput);
			printf("--low_value_tests=true/false, low value rand generate for tests, default = %d \n \n", low_value_tests);
			return 0;
        case 't':
	        test_case=atoi(optarg);
			printf("set test_case=%d \n", test_case);
			// if( sizeof(conv_Hin)/sizeof(conv_Hin[0])> test_case && optarg =!NULL){
            //     printf("set test_case=%d \n", test_case);
			// }else{
            //     printf("set test_case error, test_case shoulf be in [0, %d] \n", sizeof(conv_Hin)/sizeof(conv_Hin[0]));
			// }
			break;	
        case 'f':
	        freqset = atoi(optarg);
			printf("set freqset=%d \n", freqset);
			break;
	    case 'd':
	        divset = atoi(optarg);
			printf("set divset=%d \n", divset);
			break;
	    case 'b':
	        divset_pad = atoi(optarg);
			printf("set divset_pad=%d \n", divset_pad);
			break;
	    case 'c':
	   	    cycle_num = atoi(optarg);
			printf("set cycle_num=%d \n", cycle_num);
			break;
		case 'o':
	   	    ouput_error = atoi(optarg);
			printf("set ouput_error=%d \n", ouput_error);
			break;
		case 'p':
	   	    print_ouput = atoi(optarg);
			printf("set print_ouput=%d \n", print_ouput);
			break;
		case 'l':
	   	    low_value_tests = atoi(optarg);
			printf("set low_value_tests=%d \n", low_value_tests);
			break;
       default:
            printf("default valud: test_case=%d, freqset=%d, divset=%d,  divset_pad=%d, cycle_num=%d\n",
			                  test_case, freqset, divset, divset_pad, cycle_num);
            printf("default valud: ouputerror=%d, printouput=%d\n",
			                  ouput_error, print_ouput);
							  
        }
    }  

//Initial
    FPGA_Init();
    printf("Hello VSC!\r\n");

    printf("InitConfig...\n");

	conv_cfg *cfg = (struct conv_cfg*)calloc(1, sizeof(struct conv_cfg));
	if (cfg == NULL)
	{
	    printf("cfg malloc with len : %ld error!\n",sizeof(struct conv_cfg));
	    return EXIT_FAILURE;
	}else{printf("cfg malloc with len : %ld successful!\n",sizeof(struct conv_cfg));}

	InitConfig(cfg, 
	                conv_CHin[test_case],     conv_Hin[test_case],       conv_Win[test_case], conv_CHout[test_case],     conv_Ky[test_case], conv_Kx[test_case], 
                       pad_up[test_case],     pad_left[test_case],             Sx[test_case],         Sy[test_case], conv_wt_bit[test_case],
			     conv_out_bit[test_case], conv_relu_en[test_case], conv_out_scale[test_case], cycle_num, freqset, divset, divset_pad);
    
	ShowConfig(cfg);

//calloc rand date
    int len_rand_feature=cfg->CHin*cfg->Hin*cfg->Win;
    int8_t *feature_rand = (int8_t *)calloc(len_rand_feature,sizeof(int8_t));
	if (feature_rand == NULL){
	    printf("feature_rand malloc with len : %d error!\n",len_rand_feature);
	    return EXIT_FAILURE;
	}else{printf("feature_rand malloc with len : %d successful!\n",len_rand_feature);}

    int len_rand_weight=cfg->Ky*cfg->Kx*cfg->CHin*cfg->CHout;
	int8_t *weight_rand  = (int8_t *)calloc(len_rand_weight,sizeof(int8_t));
	if (weight_rand == NULL){
	    printf("weight_rand malloc with len : %d error!\n",len_rand_weight);
	    return EXIT_FAILURE;
	}else{printf("weight_rand malloc with len : %d successful!\n",len_rand_weight);}

//calloc Reshape data
	int32_t slice_output_pixels = ((cfg->Wout*cfg->Hout+Tout-1)/Tout);
	int32_t len_re_feature = cfg->slice_of_CHout_L0*slice_output_pixels*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
    uint32_t *re_feature = (uint32_t *)calloc(len_re_feature,sizeof(uint32_t));
	if (re_feature == NULL){
		    printf("re_feature malloc with len : %d error!\n",len_re_feature);
		    return EXIT_FAILURE;
		}else{printf("re_feature malloc with len : %d successful! at %p \n",len_re_feature, re_feature);}
 
    int32_t len_re_weight  = len_re_feature;
	uint32_t *re_weight  = (uint32_t *)calloc(len_re_weight,sizeof(uint32_t));
	if (re_weight == NULL){
	    printf("re_weight malloc with len : %d error!\n",len_re_feature);
	    return EXIT_FAILURE;
	}else{printf("re_weight malloc with len : %d successful! at %p \n",len_re_feature, re_weight);}

//calloc Reshape_conv_solft 
	int32_t len_Reshape_conv_solft = cfg->CHout*cfg->Wout*cfg->Hout;
	int8_t *Reshape_conv_solft = (int8_t *)calloc(len_Reshape_conv_solft,sizeof(int8_t));
	if (Reshape_conv_solft == NULL){
	    printf("Reshape_conv_solft malloc with len : %d error!\n",len_Reshape_conv_solft);
	    return EXIT_FAILURE;
	}else{printf("Reshape_conv_solft malloc with len : %d successful!\n",len_Reshape_conv_solft);}

//calloc conv_solft 
	int32_t len_conv_solft = cfg->CHout*cfg->Wout*cfg->Hout;
	int8_t *conv_solft = (int8_t *)calloc(len_conv_solft,sizeof(int8_t));
	if (conv_solft == NULL){
	    printf("conv_solft malloc with len : %d error!\n",len_conv_solft);
	    return EXIT_FAILURE;
	}else{printf("conv_solft malloc with len : %d successful!\n",len_conv_solft);}

//calloc cmd_cfg 
	uint32_t *cmd_cfg = (uint32_t *)calloc(7,sizeof(uint32_t));
	if (cmd_cfg == NULL){
	    printf("cmd_cfg malloc with len : %d error!\n",7);
	    return EXIT_FAILURE;
	}else{
		printf("cmd_cfg malloc with len : %d successful !\n",7);
	}

//calloc right_reg 
	uint8_t *addr_reg = (uint8_t *)calloc(13,sizeof(uint8_t));
	if (addr_reg == NULL){
	    printf("addr_reg malloc with len : %d error!\n",13);
	    return EXIT_FAILURE;
	}else{
		printf("addr_reg malloc with len : %d successful!\n",13);
	}

//calloc chip_out_result 
	int32_t len_chip_out_result = cfg->CHout*cfg->Wout*cfg->Hout ;
	int8_t *chip_out_result=(int8_t *)calloc(len_chip_out_result,sizeof(int8_t));
	if (chip_out_result == NULL)
	{
	    printf("chip_out_result malloc with len : %d error!\n",len_chip_out_result);
	    return EXIT_FAILURE;
	}else
	{
		printf("chip_out_result malloc with len : %d successful !\n",len_chip_out_result);
	}


//Generator rand date
    Rand_Gen(cfg ,feature_rand, len_rand_feature, weight_rand, len_rand_weight,low_value_tests);

//RunConv_Reshape	
	printf("Start Reshape Date\n");
	RunConv_Reshape(cfg, feature_rand, weight_rand, re_feature, len_re_feature, re_weight, len_re_weight);
	//printf("feature[%d]=%d.......................... \n", 593, re_feature[593]);

//Start Reshape conv_soft
	printf("Start Reshape conv_soft \n");
	Reshape_conv_soft(cfg, feature_rand, weight_rand, Reshape_conv_solft);

//Start conv_soft
	printf("Start conv_soft \n");
	conv_soft(cfg, feature_rand, weight_rand, conv_solft);
	printf("Finish conv_soft \n");

//using AXI dirct mode wwith Xil_Out32(AXI_MODEL,1);
	Xil_Out32(AXI_MODEL,0);
//set model sel
    Xil_Out32(MODE_SEL,0);
	printf("model_sel = %d \n", Xil_In32(MODE_SEL));

//send featue to FPGA BRAM buffer
    Write_feature(cfg, re_feature, WRITE_DAT);
	printf("send input re_feature addr= %d \n", Xil_In32(WRITE_DAT)>>16);

//send weight to FPGA BRAM buffer
	Write_weight (cfg, re_weight , WRITE_WET);
	printf("send input re_weight addr= %d \n", Xil_In32(WRITE_WET)>>16);

//Reset system Before config on-chip clock
	Xil_Out32(RESET_VSC,0);
    printf("Reset System before config on-chip clock......\n");
    printf("Finish System before config on-chip clock!!! \n");
    Xil_Out32(RESET_VSC,1);

//Config on-chip clock
	config_on_chip_clock(cfg, CFG_REG);
    //read_all_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL);

//Reset system after config on-chip clock
	Xil_Out32(RESET_VSC,0);
    printf("Reset System after config on-chip clock......\n");
    printf("Finish System after config on-chip clock!!! \n");
    Xil_Out32(RESET_VSC,1);
    //read_all_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL);

//config clock div in FPGA
//	Xil_Out32(CLK_DVI_NUM,0);
//	printf("Finish CLK_DVI_NUM \n");
//config clock div in FPGA
//	Xil_Out32(CLK_DVI_NUM,2);

//Config on-chip register......
	printf("Config on-chip register......\n");
    config_on_chip_register(cfg, CFG_REG, cmd_cfg);
	printf("Config on-chip register finish! \n");

//on-chip register Validattion......
	check_on_chip_register(cfg, CFG_REG, GET_CFG_REG, MODE_SEL, addr_reg);

// //start vsc
// 	//read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 17);
//     start_vsc(START_VSC);

// // 	for(int i=0; i<1000; i++){
// // 	uint8_t l_wr_feature_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 13);
// //  uint8_t h_wr_feature_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 14);
// // 	uint8_t l_wr_weight_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 15);
// // 	uint8_t h_wr_weight_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 16);
// // 	uint16_t wr_feature_addr=l_wr_feature_addr+(h_wr_feature_addr<<8);
// // 	uint16_t wr_weight_addr = l_wr_weight_addr+(h_wr_weight_addr<<8);
// // 	printf("wr_feature_addr = %d; wr_weight_addr = %d \n", wr_feature_addr, wr_weight_addr);
// // 	}
// // for(int i=0; i<1000; i++){
// //     printf("chip state= %x \n", read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 17));
// // }

// // 	for(int i=0; i<1000; i++){
// // 	uint8_t l_rd_feature_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 18);
// //     uint8_t h_rd_feature_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 19);
// // 	uint8_t l_rd_weight_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 20);
// // 	uint8_t h_rd_weight_addr=read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 21);
// // 	uint16_t rd_feature_addr=l_rd_feature_addr+(h_rd_feature_addr<<8);
// // 	uint16_t rd_weight_addr = l_rd_weight_addr+(h_rd_weight_addr<<8);
// // 	printf("rd_feature_addr = %d; rd_weight_addr = %d \n", rd_feature_addr, rd_weight_addr);
// // 	}
// //
    

// //Waiting to computing finish & Read output result
    char *conv_solf_name = "conv_solf";
    char *vcs_out_name = "vsc_out";
// if(waiting_finish(GET_OUT_VALID, GET_DONE)){
//     printf("Start Read output result!! \n");
// 	read_chip_out_result(cfg, chip_out_result, READ_ADDR,  READ_ODAT) ;
// //chip_out_result compare with conv solf output result	
// 	printf("Start compare result!! \n");
//     compare_result(cfg, vcs_out_name, chip_out_result,conv_solf_name, conv_solft, ouput_error, print_ouput);
// }else{
// 	printf("Wating finsh fial !!!");
// }

//Re_conv_solf compare with conv solf output result
    char *Re_conv_solf_name = "Re_conv_solf";
    compare_result(cfg, Re_conv_solf_name, Reshape_conv_solft, conv_solf_name, conv_solft, ouput_error, false);

    free(feature_rand); //printf("free (feature_rand) \n");
    feature_rand=NULL;
    free(weight_rand);  //printf("free (weight_rand) \n");
    weight_rand=NULL;  
    free(re_feature);  //printf("free (re_feature) \n");
    re_feature=NULL;
    free(re_weight);  //printf("free (re_weight) \n");
    re_weight=NULL;
    free(Reshape_conv_solft); //printf("free (Reshape_conv_solft) \n");
    Reshape_conv_solft=NULL;
    free(conv_solft);  //printf("free (conv_solft) \n");
    conv_solft=NULL;
    free(cmd_cfg);  //printf("free (cmd_cfg) \n");
    cmd_cfg=NULL;
    free(addr_reg); //printf("free (addr_reg) \n");
    addr_reg=NULL;
    free(chip_out_result); //printf("free (chip_out_result) \n");
    chip_out_result=NULL;

   return 0;

}



