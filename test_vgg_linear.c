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
#include "model_structure.h"
//#include "typical_structure.h
//#include "rand_structure.h"

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
	conv_cfg *cfg = (struct conv_cfg*)calloc(1, sizeof(struct conv_cfg));
	InitConfig(cfg, 
	                conv_CHin[test_case],     conv_Hin[test_case],       conv_Win[test_case], conv_CHout[test_case],     conv_Ky[test_case], conv_Kx[test_case], 
                       pad_up[test_case],     pad_left[test_case],             Sx[test_case],         Sy[test_case], conv_wt_bit[test_case],
			     conv_out_bit[test_case], conv_relu_en[test_case], conv_out_scale[test_case], cycle_num, freqset, divset, divset_pad);	
    ShowConfig(cfg);

//calloc vgg date
    int len_read_feature=cfg->CHin*cfg->Hin*cfg->Win;
    int8_t *read_feature = (int8_t *)calloc(len_read_feature,sizeof(int8_t));
	if (read_feature == NULL){
	    printf("read_feature malloc with len : %d error!\n",len_read_feature);
	    return EXIT_FAILURE;
	}else{printf("read_feature malloc with len : %d successful!\n",len_read_feature);}

    int len_read_weight=cfg->Ky*cfg->Kx*cfg->CHin*cfg->CHout;
	int8_t *read_weight  = (int8_t *)calloc(len_read_weight,sizeof(int8_t));
	if (read_weight == NULL){
	    printf("read_weight malloc with len : %d error!\n",len_read_weight);
	    return EXIT_FAILURE;
	}else{printf("read_weight malloc with len : %d successful!\n",len_read_weight);}
    
//calloc Reshape data
	int32_t slice_output_pixels = ((cfg->Wout*cfg->Hout+Tout-1)/Tout);

	uint32_t div_KxKyTout = WT_SRAM_DEPTH/(cfg->Ky*cfg->Kx*Tout);
    uint32_t div_slice_of_CHin_L0 = (cfg->slice_of_CHin_L0+div_KxKyTout-1)/div_KxKyTout;

	uint32_t div_KxKyCin;
    if(div_slice_of_CHin_L0 >1)
	{
		div_KxKyCin=1;
         
	}
	else if (div_slice_of_CHin_L0 =1){
        div_KxKyCin = WT_SRAM_DEPTH/(cfg->Ky*cfg->Kx*Tout*cfg->slice_of_CHin_L0);
	}
	else{
		printf("Error with div_slice_of_CHin_L0 < 1 !!!\n");
		return EXIT_FAILURE;
	}

    uint32_t div_out_pixels = (slice_output_pixels+div_KxKyCin-1)/div_KxKyCin;
    
	//uint32_t div_test_num = cfg->slice_of_CHout_L0 * div_out_pixels;
	//int32_t len_re_feature = div_test_num*div_KxKyCin*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
	
	uint32_t div_test_num = cfg->slice_of_CHout_L0*div_out_pixels;
	uint32_t div_len_chin = div_KxKyTout*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
	uint32_t div_len= div_KxKyCin*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
	int32_t len_re_feature = div_test_num*div_len;
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
//Read date
    // char *feature_file = "./test_model/conv_layer0_8bit_act_s=-3.0.txt";
	// char *weight_file  = "./test_model/conv_layer0_8bit_weight_s=-6.0.txt";
    char *feature_file = feature_filelist[test_case];//"./test_model/conv_layer1_4bit_act_s=-3.0.txt";
	char *weight_file  = weight_filelist[test_case];//"./test_model/conv_layer1_4bit_weight_s=-6.0.txt";
    
	uint32_t in_scale = conv_in_scale[test_case];
    uint32_t wt_scale = conv_wt_scale[test_case];
    //Read_data(cfg , feature_file, read_feature, len_read_feature, weight_file, read_weight, len_read_weight, in_scale, wt_scale);
	Read_bin_data(cfg , feature_file, read_feature, len_read_feature, weight_file, read_weight, len_read_weight, in_scale, wt_scale);
//RunConv_Reshape	
	printf("Start Reshape Date\n");
	conv_div(cfg, read_feature, read_weight, re_feature, len_re_feature, re_weight, len_re_feature);
	//RunConv_Reshape(cfg, read_feature, read_weight, re_feature, len_re_feature, re_weight, len_re_feature);
	//printf("feature[%d]=%d.......................... \n", 593, re_feature[593]);

//Start Reshape conv_soft
	//printf("Start Reshape conv_soft \n");
	//Reshape_conv_soft(cfg, read_feature, read_weight, Reshape_conv_solft);

//Start conv_soft
	// printf("Start conv_soft \n");
	// conv_soft(cfg, read_feature, read_weight, conv_solft);
	// printf("Finish conv_soft \n");

    uint32_t conv_Hin_div        =1                          ;
    uint32_t conv_Win_div        =Tout*div_KxKyCin           ;
    uint32_t conv_CHin_div       =cfg->Tin_L0*cfg->slice_of_CHin_L0;
    uint32_t conv_CHout_div      =Tout                       ;
    uint32_t conv_Ky_div         =conv_Ky[test_case]         ;
    uint32_t conv_Kx_div         =conv_Kx[test_case]         ;
    uint32_t Sx_div              =Sx[test_case]              ;
    uint32_t Sy_div              =Sy[test_case]              ;
    uint32_t pad_left_div        =pad_left[test_case]        ;
    uint32_t pad_right_div       =pad_right[test_case]       ;
    uint32_t pad_up_div          =pad_up[test_case]          ;
    uint32_t pad_down_div        =pad_down[test_case]        ;

    //structure of VGG16:          0re of VGG16:      ;
    uint32_t conv_wt_scale_div   =conv_wt_scale[test_case]   ;
    uint32_t conv_in_scale_div   =conv_in_scale[test_case]   ;
    uint32_t conv_out_scale_div  =conv_out_scale[test_case]  ;
    uint32_t conv_wt_bit_div     =conv_wt_bit[test_case]     ;
    uint32_t conv_in_bit_div     =conv_in_bit[test_case]     ;
    uint32_t conv_out_bit_div    =conv_out_bit[test_case]    ;
    uint32_t conv_relu_en_div    =conv_relu_en[test_case]    ;

    //Reset system Before config on-chip clock
    	Xil_Out32(RESET_VSC,0);
        //printf("Reset System before config on-chip clock......\n");
        //printf("Finish System before config on-chip clock!!! \n");
        Xil_Out32(RESET_VSC,1);
    
    //Config on-chip clock
    	config_on_chip_clock(cfg, CFG_REG);
        //read_all_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL);
    
    //Reset system after config on-chip clock
    	Xil_Out32(RESET_VSC,0);
        //printf("Reset System after config on-chip clock......\n");
        //printf("Finish System after config on-chip clock!!! \n");
        Xil_Out32(RESET_VSC,1);

    for(uint32_t test_i =0; test_i <  cfg->slice_of_CHout_L0; test_i++)
	{
		//printf("test_i / div_test_num = %d /%d Start !!!\n", test_i, cfg->slice_of_CHout_L0);
	for(uint32_t test_j =0; test_j < div_out_pixels; test_j++)
	{
    for(uint32_t test_k =0; test_k < div_slice_of_CHin_L0; test_k++)
	{
		if(div_slice_of_CHin_L0>1)
		{
			if(test_k==(div_slice_of_CHin_L0-1))
			{
				conv_CHin_div = cfg->CHin - test_k*div_KxKyTout*cfg->Tin_L0;
			}
			else
			{
                conv_CHin_div = div_KxKyTout*cfg->Tin_L0;
			}
            
			conv_Win_div  = Tout;
		}
		else
		{
            conv_CHin_div = cfg->Tin_L0*cfg->slice_of_CHin_L0;
            if(test_j==(div_out_pixels-1))
		    {
                conv_Win_div= (cfg->Hout*cfg->Wout) - test_j*(Tout*div_KxKyCin);
		    	//printf("CHout = %d, Wout= %d, div_KxKyCin= %d\n", cfg->CHout, cfg->Wout, div_KxKyCin);
		    	//printf("conv_Win_div = %d \n", conv_Win_div);
		    }
		    else
		    {
                conv_Win_div=Tout*div_KxKyCin;
		    }

		}
		    //printf("test_j = %d, div_out_pixels = %d, \n", test_j,div_out_pixels);
        	conv_cfg *cfg_div = (struct conv_cfg*)calloc(1, sizeof(struct conv_cfg));
        	if (cfg_div == NULL)
        	{
        	    printf("cfg malloc with len : %ld error!\n",sizeof(struct conv_cfg));
        	    return EXIT_FAILURE;
        	}else{
				//printf("cfg malloc with len : %ld successful!\n",sizeof(struct conv_cfg));
			}
            
 
			//printf("InitConfig...\n");
        	InitConfig(cfg_div, 
        	                conv_CHin_div,     conv_Hin_div,       conv_Win_div, conv_CHout_div,     conv_Ky_div, conv_Kx_div, 
                               pad_up_div,     pad_left_div,             Sx_div,         Sy_div, conv_wt_bit_div,
        			     conv_out_bit_div, conv_relu_en_div, conv_out_scale_div, cycle_num, freqset, divset, divset_pad);
            
        	//ShowConfig(cfg_div);
            
        	if (cfg_div->input_data_num > WT_SRAM_DEPTH) {
        		printf(" error! weight/input SRAM not enough! \n");
        	    return EXIT_FAILURE;
        	}
        
        	if (cfg_div->output_data_num > OUTPUT_SRAM_DEPTH) {
        		printf(" error! output SRAM not enough! \n");
        	    return EXIT_FAILURE;
        	}
        
        
        //calloc cmd_cfg 
        	uint32_t *cmd_cfg = (uint32_t *)calloc(7,sizeof(uint32_t));
        	if (cmd_cfg == NULL){
        	    printf("cmd_cfg malloc with len : %d error!\n",7);
        	    return EXIT_FAILURE;
        	}else{
        		//printf("cmd_cfg malloc with len : %d successful !\n",7);
        	}
        
        //calloc right_reg 
        	uint8_t *addr_reg = (uint8_t *)calloc(13,sizeof(uint8_t));
        	if (addr_reg == NULL){
        	    printf("addr_reg malloc with len : %d error!\n",13);
        	    return EXIT_FAILURE;
        	}else{
        		//printf("addr_reg malloc with len : %d successful!\n",13);
        	}
        
        //using AXI dirct mode wwith Xil_Out32(AXI_MODEL,1);
        	Xil_Out32(AXI_MODEL,0);
        //set model sel
            Xil_Out32(MODE_SEL,0);
        	//printf("model_sel = %d \n", Xil_In32(MODE_SEL));
        
        //send featue to FPGA BRAM buffer
		uint32_t skip_len = div_len*(test_j+test_i*div_out_pixels)+test_k*div_len_chin ;
		//printf("skip_len = %d \n", skip_len);
		//uint32_t skip_len = cfg_div->input_data_num*16;
            Write_feature(cfg_div, &re_feature[skip_len], WRITE_DAT);

        	//printf("send input re_feature addr= %d \n", Xil_In32(WRITE_DAT)>>16);
        
        //send weight to FPGA BRAM buffer
        	Write_weight (cfg_div, &re_weight[skip_len] , WRITE_WET);

        	//printf("send input re_weight addr= %d \n", Xil_In32(WRITE_WET)>>16);
        
        //Reset system Before config on-chip clock
        	//Xil_Out32(RESET_VSC,0);

            //printf("Reset System before config on-chip clock......\n");
            //printf("Finish System before config on-chip clock!!! \n");
            //Xil_Out32(RESET_VSC,1);

        
        //Config on-chip clock
        	//config_on_chip_clock(cfg_div, CFG_REG);

            //read_all_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL);
        
        //Reset system after config on-chip clock
        	Xil_Out32(RESET_VSC,0);

            //printf("Reset System after config on-chip clock......\n");
            //printf("Finish System after config on-chip clock!!! \n");
            Xil_Out32(RESET_VSC,1);

            //read_all_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL);
        
        //Config on-chip register......
        	//printf("Config on-chip register......\n");
            config_on_chip_register(cfg_div, CFG_REG, cmd_cfg);

        	//printf("Config on-chip register finish! \n");
        
        //on-chip register Validattion......
        	//check_on_chip_register(cfg_div, CFG_REG, GET_CFG_REG, MODE_SEL, addr_reg);
        
        // //start vsc
        	//read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 17);
            start_vsc(START_VSC);

        
        //Waiting to computing finish & Read output result
        	//int32_t len_chip_out_result_div = cfg_div->CHout*cfg_div->Wout*cfg_div->Hout ;
            if(waiting_finish(GET_OUT_VALID, GET_DONE)){
                //printf("Start Read output result!! \n");
            	read_chip_out_result_div_linera(cfg_div, cfg, chip_out_result, READ_ADDR,  READ_ODAT, test_i, test_j, div_KxKyCin, test_k) ;
            //chip_out_result compare with conv solf output result	
            }else{
            	printf("Wating finsh fial !!!");
            }
            free(cmd_cfg);  //printf("free (cmd_cfg) \n");
            cmd_cfg=NULL;
            free(addr_reg); //printf("free (addr_reg) \n");
            addr_reg=NULL;
			printf("test_i / div_test_num = %d / %d, test_j / div_out_pixels = %d / %d, test_k / div_slice_of_CHin_L0 = %d /%d Finish !!!\n", test_i, cfg->slice_of_CHout_L0, test_j, div_out_pixels, test_k, div_slice_of_CHin_L0);
	}
	}
			//printf("test_i / div_test_num = %d /%d Finish !!!\n", test_i, cfg->slice_of_CHout_L0);
    }

    printf("All computation finsh !!!\n");
    char *conv_solf_name = "conv_soft";
    char *vcs_out_name = "vsc_out";
//chip_out_result compare with conv solf output result	
    printf("Start compare result!! \n");
	//printf("chip_out_result[%d] = %d \n", 16, chip_out_result[16]);
    compare_result(cfg, vcs_out_name, chip_out_result,conv_solf_name, conv_solft, ouput_error, print_ouput);
//Re_conv_solf compare with conv solf output result
    //char *Re_conv_solf_name = "Re_conv_solf";
    //compare_result(cfg, Re_conv_solf_name, Reshape_conv_solft, conv_solf_name, conv_solft, ouput_error, false);

    free(read_feature); //printf("free (feature_rand) \n");
    read_feature=NULL;
    free(read_weight);  //printf("free (weight_rand) \n");
    read_weight=NULL;  
    free(re_feature);  //printf("free (re_feature) \n");
    re_feature=NULL;
    free(re_weight);  //printf("free (re_weight) \n");
    re_weight=NULL;
    free(Reshape_conv_solft); //printf("free (Reshape_conv_solft) \n");
    Reshape_conv_solft=NULL;
    free(conv_solft);  //printf("free (conv_solft) \n");
    conv_solft=NULL;
    free(chip_out_result); //printf("free (chip_out_result) \n");
    chip_out_result=NULL;

   return 0;

}



