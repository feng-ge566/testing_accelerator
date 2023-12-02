#include <sys/mman.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include "cnn_defines.h"
#include "basic.h"
#include <unistd.h>
void *axi_addr_base;

void FPGA_Init()
{
    int fd;
	fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fd == -1)
		printf("Error: Can't open /dev/mem\n");

	axi_addr_base = mmap(0, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xA0000000);
	if (axi_addr_base == NULL)
		printf("Error: axi_addr_base mmap fail\n");

	// mem_map_base = mmap(0, 0x100000000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x400000000);
	// if (mem_map_base == NULL)
	// 	printf("Error: mem_base mmap fail\n");
	
	printf("**** axi_addr_base=%p\n", axi_addr_base);
	//printf("**** mem_map_base=%d\n", mem_map_base);
	
	printf("FPGA Init Done\n");
}
void Xil_Out32(int addr, uint32_t data)
{
	*(uint32_t*)(((long int)axi_addr_base) + (addr << 2)) = data;
}

uint32_t Xil_In32(int addr)
{
    uint32_t rt = *(uint32_t*)(((long int)axi_addr_base) + (addr << 2));
	return rt;
}
void InitConfig(conv_cfg *cfg, 
		uint32_t CHin, uint32_t Hin, uint32_t Win, uint32_t CHout, uint32_t Ky, uint32_t Kx,
		uint32_t Py  , uint32_t Px , uint32_t Sy , uint32_t Sx   , uint32_t DAT_DW_L0,
		uint32_t DAT_DW_L1, uint32_t  RELU_EN_L0, uint32_t OUT_Shift,uint32_t practical_log2_cycle,
        uint8_t freqset, uint8_t divset, uint8_t divset_pad)
{
    //conv_cfg *cfg = (conv_cfg *)malloc(sizeof(conv_cfg));
    cfg->CHin  = CHin ; 
    cfg->Hin   = Hin  ; 
    cfg->Win   = Win  ; 
    cfg->CHout = CHout;
    cfg->Ky    = Ky   ; 
    cfg->Kx    = Kx   ; 
    cfg->Py    = Py   ; 
    cfg->Px    = Px   ; 
    cfg->Sy    = Sy   ; 
    cfg->Sx    = Sx   ;

    cfg->DAT_DW_L0=DAT_DW_L0;
    cfg->DAT_DW_L1=DAT_DW_L1;
    cfg->OUT_Shift=OUT_Shift;
    
    cfg->RELU_EN_L0=RELU_EN_L0;
    cfg->practical_log2_cycle=practical_log2_cycle;

    cfg->WT_DW_L0  =cfg->DAT_DW_L0;

	cfg->Tin_L0=(cfg->DAT_DW_L0 == 8) ? base_Tin/2 :base_Tin*MAX_DW/cfg->DAT_DW_L0;
    cfg->Wout  = ((cfg->Win + 2*cfg->Px - cfg->Kx)/cfg->Sx + 1);
    cfg->Hout  = ((cfg->Hin + 2*cfg->Py - cfg->Ky)/cfg->Sy + 1);
    //cfg->CHout = Tout;

    cfg->dw      = cfg->DAT_DW_L0;
    cfg->dw_num  = MAX_DW / cfg->dw;
    cfg->Tin_num = (cfg->dw == 8) ? Tin_int8 : base_Tin;
    cfg->Tin_l0  = cfg->Tin_num * cfg->dw_num;


    cfg->slice_of_Wout_x_Hout    = (cfg->Wout*cfg->Hout+Tout-1)/Tout;
    cfg->total_output_pixels     = (cfg->slice_of_Wout_x_Hout*Tout);
    cfg->slice_of_CHout_L0       = ((cfg->CHout+Tout-1)/Tout);
    cfg->slice_of_CHin_L0        = ((cfg->CHin+cfg->Tin_l0-1)/cfg->Tin_l0);
    cfg->slice_of_Tin_div_Tout_L0= ((cfg->Tin_L0+Tout-1)/Tout);

    cfg->Tin_factor=cfg->DAT_DW_L0==8?1:(16/cfg->DAT_DW_L0);

    cfg->out_data_width=cfg->DAT_DW_L1-1;

    cfg->acc_times_minus1=cfg->slice_of_CHin_L0*cfg->Kx*cfg->Ky-1;

    cfg->pixel_group_num_minus1=cfg->slice_of_Wout_x_Hout-1;

    cfg->input_data_num     = cfg->total_output_pixels*cfg->Kx*cfg->Ky*cfg->slice_of_CHin_L0;

    cfg->output_data_num = cfg->total_output_pixels;

    cfg->data_shape= cfg->slice_of_CHout_L0*cfg->slice_of_Wout_x_Hout*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/2;

    cfg->div_num=2;
    cfg->freqset=freqset;
    cfg->divset=divset;
    cfg->divset_pad=divset_pad;
}
int in_index(struct conv_cfg *cfg, uint32_t i, uint32_t j, uint32_t k) {
    return cfg->Hin*cfg->Win*i + cfg->Win*j + k;
}

int wt_index(struct conv_cfg *cfg, uint32_t i, uint32_t j, uint32_t k, uint32_t l) {
    return cfg->CHin*cfg->Ky*cfg->Kx*i + cfg->Ky*cfg->Kx*j + cfg->Kx*k + l;
}

int out_index(struct conv_cfg *cfg, uint32_t i, uint32_t j, uint32_t k) {
    return cfg->Hout*cfg->Wout*i + cfg->Wout*j + k;
}

void ShowConfig(conv_cfg *cfg) {

	printf("\n-------------CONFIG--------------\n");
    printf("Hin: %d, Win: %d, CHin: %d, CHout: %d\n", 
            cfg->Hin, cfg->Win, cfg->CHin, cfg->CHout);
    printf("Ky: %d, Kx: %d, Py: %d, Px: %d, Sy: %d, Sx: %d\n", 
            cfg->Ky, cfg->Kx, cfg->Py, cfg->Px, cfg->Sy, cfg->Sx);
    printf("DAT_DW_L0: %d, DW_num: %d, Tin: %d, Tin_num: %d\n", 
            cfg->DAT_DW_L0, cfg->dw_num, cfg->Tin_l0, cfg->Tin_num);
    printf("S_CHout: %d, S_CHin: %d, T_pixel: %d\n",
            cfg->slice_of_CHout_L0, cfg->slice_of_CHin_L0, cfg->total_output_pixels);
    printf("input_data_num: %d,  output_dat_num: %d\n",
            cfg->input_data_num, cfg->output_data_num);
    printf("Tin_l0: %d,  dw_num: %d, Tin_num: %d, dw: %d\n",
            cfg->Tin_l0, cfg->dw_num, cfg->Tin_num, cfg->dw);
            printf("Relu_en = %d \n", cfg->RELU_EN_L0);
    printf("-------------EndCFG--------------\n");
}

void conv_div(conv_cfg *cfg, int8_t *addr_feature, int8_t *addr_weight,
                                    uint32_t *BASE_DAT, int32_t len_re_feature,
                                    uint32_t *BASE_WT, int32_t len_re_weight)
{
	int8_t in_scale=0;
	int8_t wt_scale=0;
	int8_t out_scale=cfg->OUT_Shift;

	int8_t feat_in;
	int8_t weight;
	uint16_t tp_dt, tp_wt;

	uint8_t tpp_dt, tpp_wt;
    int32_t h, w, row, col;
    uint32_t Chin_tmp=0;
    int32_t cnt_dat=0;
    int32_t cnt_wt=0;
    int32_t cnt_t=0;
    int32_t cnt_tt = 0;
    int32_t slice_output_pixels = ((cfg->Wout*cfg->Hout+Tout-1)/Tout);
    int32_t all_for=len_re_feature; //cfg->slice_of_CHout_L0*slice_output_pixels*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
    uint32_t prt_dt;
    //**********************************************div********************************************//

    // uint32_t div_KxKyTout = WT_SRAM_DEPTH/(cfg->Ky*cfg->Kx*Tout);
    // uint32_t div_slice_of_CHin_L0 = (cfg->slice_of_CHin_L0+div_KxKyTout-1)/div_KxKyTout;

    // uint32_t div_KxKyCin = WT_SRAM_DEPTH/(cfg->Ky*cfg->Kx*Tout*cfg->slice_of_CHin_L0);
    // uint32_t div_out_pixels = (slice_output_pixels+div_KxKyCin-1)/div_KxKyCin;


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
	}
    uint32_t div_out_pixels = (slice_output_pixels+div_KxKyCin-1)/div_KxKyCin;

    //printf("BASE_DAT = %p \n", BASE_DAT);
    //printf("BASE_WT = %p \n", BASE_WT);
	//printf("%d=%dx%dx%dx%dx%dx%dx%d \n",all_for,cfg->slice_of_CHout_L0,slice_output_pixels,cfg->slice_of_CHin_L0, cfg->Ky,cfg->Kx,Tout, cfg->Tin_num/cfg->div_num);
    printf("cnt_dat/all_for_dat = %d/%d \n", cnt_t,all_for);
    //reshape feature
        for(int chout=0;chout<cfg->slice_of_CHout_L0;chout++) {
            //for(int p=0;p<slice_output_pixels;p++) {
             for(int div_p=0;div_p<div_out_pixels;div_p++) {

                for(int div_c0=0;div_c0<div_KxKyCin;div_c0++) {
                    int p=div_p*div_KxKyCin+div_c0;
                //(kx*ky*Tout*slice_of_CHin_L0)<2304
                //slice_of_CHin_L0<16
                //(2304/cfg->Ky*cfg->Kx*Tout*cfg->slice_of_CHin_L0  cfg->slice_of_Wout_x_Hout*Tout);
                for(int chin=0;chin<cfg->slice_of_CHin_L0;chin++) {
                //for(int cyc0=0;cyc0<div_slice_of_CHin_L0;cyc0++) {
                                    //kx*ky<144
                    //for(int div0=0; div0<div_KxKyTout; div0++){int chin= div0 + cyc0*div_KxKyTout; }
                    for(int ky=0;ky<cfg->Ky;ky++){
                        for(int kx=0;kx<cfg->Kx;kx++) {
                            for(int pp=0;pp<Tout;pp++) {
                                h=(p*Tout+pp) / cfg->Wout;
                                w=(p*Tout+pp) % cfg->Wout;
                                row=h*cfg->Sy - cfg->Py + ky;
                                col=w*cfg->Sx - cfg->Px + kx;
                                //for(int tout=0;tout<Tout;tout++) {
                                	for(int tin_div=0; tin_div < cfg->Tin_num/cfg->div_num; tin_div++)
                                    {
                                        tp_dt = 0;
                                        for(int tin=0;tin<cfg->div_num;tin++)
                                        {
                                        	tpp_dt =0;
                                            for (int dw=0; dw < cfg->dw_num; dw++)
                                            {
                                            	Chin_tmp=(chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw);
                                                if (row<0 || col<0 || row >= cfg->Hin || col >= cfg->Win ||  Chin_tmp>= cfg->CHin)
                                                {
                                                	tpp_dt = tpp_dt+ (0 << (cfg->dw * dw));
                                                }

                                                else
                                                {
                                                	      feat_in=addr_feature[in_index(cfg, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, row, col)];
									    			      if(cfg->DAT_DW_L0==1)
									    				  {
									    			    	  tpp_dt = tpp_dt+((feat_in&0x01)<< (cfg->dw * dw));
                                                	          //tpp_dt = tpp_dt+( (feat_in[cfg->in_index(cfg, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, row, col)]&0x01)<< (cfg->dw * dw));
                                                          }
									    				  else if(cfg->DAT_DW_L0==2)
                                                          {
                                                              tpp_dt = tpp_dt+((feat_in&0x03)<< (cfg->dw * dw));
                                                          }
                                                          else if(cfg->DAT_DW_L0==4)
                                                          {
                                                              tpp_dt = tpp_dt+((feat_in&0x0f)<< (cfg->dw * dw));
                                                          }
                                                          else if(cfg->DAT_DW_L0==8)
                                                          {
                                                              tpp_dt = tpp_dt+(feat_in << (cfg->dw * dw));
                                                          }
                                                }
                                            }
                                            tp_dt = tp_dt + (tpp_dt << tin*8);
                                        }
//                                    if(cnt_t%100==0){
//                                    					printf("tp_dt[%d] = %x \n", cnt_t, tp_dt);
//                                    					}
                                            prt_dt=cnt_dat*cfg->Tin_num/cfg->div_num+tin_div;
                                            // if(BASE_DAT[prt_dt]>BASE_DAT[all_for]|| (BASE_DAT[prt_dt])< (BASE_DAT)){
                                            //       printf("BASE_DAT[%d] out of rang ! &BASE_DAT[%d]=%p, &BASE_DAT[%d]=%p, &BASE_DAT=%p\n", prt_dt, prt_dt, &BASE_DAT[prt_dt],all_for, &BASE_DAT[all_for], BASE_DAT);
                                            // }else{}
                                            BASE_DAT[prt_dt]=tp_dt;
                                        // if(&BASE_DAT[cnt_dat*cfg->Tin_num/cfg->div_num+tin_div]>&BASE_DAT[all_for]|| (&BASE_DAT[cnt_dat*cfg->Tin_num/cfg->div_num+tin_div])< (&BASE_DAT)){
                                        //     printf("BASE_DAT[%d] out of rang !\n", cnt_dat*cfg->Tin_num/cfg->div_num+tin_div);
                                        // }else{BASE_DAT[cnt_dat*cfg->Tin_num/cfg->div_num+tin_div]=tp_dt;}
                                        
                                        //printf("BASE_DAT[%d] = %d tp_dt= %d \n",prt_dt,BASE_DAT[prt_dt],tp_dt);
                                        //printf("BASE_DAT[%d]=%d.......................... \n", 593, BASE_DAT[593]);
                                        //ASE_DAT[cnt_dat*cfg->Tin_num+tin_div*cfg->div_num]=tp_dt;
                                        cnt_t=cnt_t+1;
                                	}
                            //}
                                cnt_dat += 1;
                            }
                            //cnt_wt += Tout;
                        }
                    }
                }
                //printf("cnt_da/all_for_dat = %d/%d \n", cnt_t,all_for);
            }
        }
        }
        printf("cnt_da/all_for_dat = %d/%d \n", cnt_t,all_for);
        all_for=len_re_weight;//cfg->slice_of_CHout_L0*slice_output_pixels*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
        printf("cnt_wt/all_for_wt = %d/%d \n", cnt_tt,all_for);
        //reshape weight
        uint32_t prt_wt=0;
            for(int chout=0;chout<cfg->slice_of_CHout_L0;chout++) {
                //for(int p=0;p<slice_output_pixels;p++) {
                for(int div_p=0;div_p<div_out_pixels;div_p++) {
                for(int div_c0=0;div_c0<div_KxKyCin;div_c0++) {
                    int p=div_p*div_KxKyCin+div_c0;
                    for(int chin=0;chin<cfg->slice_of_CHin_L0;chin++) {
                        for(int ky=0;ky<cfg->Ky;ky++){
                            for(int kx=0;kx<cfg->Kx;kx++) {
                                //for(int pp=0;pp<Tout;pp++) {
//                                    h=(p*Tout+pp) / cfg->Wout;
//                                    w=(p*Tout+pp) % cfg->Wout;
//                                    row=h*cfg->Sy - cfg->Py + ky;
//                                    col=w*cfg->Sx - cfg->Px + kx;
                                    for(int tout=0;tout<Tout;tout++)
                                    {
                                    	for(int tin_div=0; tin_div < cfg->Tin_num/cfg->div_num; tin_div++)
                                        {
                                            tp_wt = 0;
                                            for(int tin=0;tin<cfg->div_num;tin++)
                                            {
                                            	tpp_wt =0;
                                                for (int dw=0; dw < cfg->dw_num; dw++)
                                                {
                                                	Chin_tmp=(chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw);
                                                    if (Chin_tmp>= cfg->CHin)
                                                    {
                                                	  tpp_wt += 0<< (cfg->dw * dw);
                                                    }
                                                    else
                                                    {
                                                	      weight=addr_weight[wt_index(cfg, chout*Tout+tout, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, ky, kx)];
										    		      if(cfg->DAT_DW_L0==1)
										    			  {
										    		    	  tpp_wt += (weight&0x01)<< (cfg->dw * dw);
                                                	           //tpp_wt += (weight[cfg->wt_index(cfg, chout*Tout+tout, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, ky, kx)]&0x01)<< (cfg->dw * dw);
                                                          }
										    			  else if(cfg->DAT_DW_L0==2)
                                                          {
                                                               tpp_wt += (weight&0x03)<< (cfg->dw * dw);
                                                          }
                                                          else if(cfg->DAT_DW_L0==4)
                                                          {
                                                                tpp_wt += (weight&0x0f)<< (cfg->dw * dw);
                                                          }
                                                          else if(cfg->DAT_DW_L0==8)
                                                          {
                                                                tpp_wt += weight << (cfg->dw * dw);
                                                          }
                                                    }

                                                }
                                                tp_wt = tp_wt+ (tpp_wt<<(tin*8));
                                             }
//                                            if(cnt_tt%100==0){
//                                                                                					printf("tp_wt[%d] = %x \n", cnt_tt, tp_wt);
//                                                                                					}
                                            prt_wt=(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div;
                                            // if(&BASE_WT[prt_wt]>&BASE_WT[all_for]|| (&BASE_WT[prt_wt])< (&BASE_WT)){
                                            //       printf("BASE_WT[%d] out of rang ! &BASE_WT[%d]=%p, &BASE_WT[%d]=%p, &BASE_WT=%p\n", prt_wt, prt_wt, &BASE_WT[prt_wt],all_for, &BASE_WT[all_for], &BASE_WT);
                                            // }else{BASE_WT[prt_wt]=tp_wt;}
                                            BASE_WT[prt_wt]=tp_wt;
                                            //printf("cnt=%d  ", cnt_tt);
                                            //test_malloc();
                                            //printf("BASE_WT[%d] = %d tp_wt= %d \n",prt_wt,BASE_WT[prt_wt],tp_wt);
                                            //printf("BASE_WT[%d] = %d, &BASE_WT[%d]= %x\n",(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div,BASE_WT[(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div], (cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div,&BASE_WT[(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div]);
                                            //printf("BASE_DAT[%d]=%d.......................... &BASE_DAT[%d]= %x\n", 593, BASE_DAT[593], 593, &BASE_DAT[593]);
                                            //BASE_WT[(cnt_wt+tout)*cfg->Tin_num+tin_div*cfg->div_num]=tp_wt;
                                            cnt_tt=cnt_tt+1;
                                    	}
                                    }
                                    //cnt_dat += 1;
                                //}
                                cnt_wt += Tout;

                            }
                        }
                    }
                    //printf("cnt_wt/all_for_wt = %d/%d \n", cnt_tt,all_for);
                }
            }
            }
            printf("cnt_wt/all_for_wt = %d/%d \n", cnt_tt,all_for);

}

void conv_soft(conv_cfg *cfg , int8_t *addr_feature,  int8_t *addr_weight, int8_t *CONV_SOLFT)
{
	int8_t in_scale=0;
	int8_t wt_scale=0;
	int8_t out_scale=0;
    int16_t shift_value;
    int8_t shift_sign;
//    int16_t out_truncate;
    shift_sign=0;//right shift
    shift_value=(in_scale+wt_scale)-out_scale;
    if(shift_value<0){
        shift_value=out_scale-(in_scale+wt_scale);
        shift_sign=1; //left shift
    }
   // out_truncate={shift_sign,shift_value}; //$display("out_truncate=%b",out_truncate);
//int32_t tp2;
int8_t tp_sat;
int8_t dat_in;
int8_t wt;
	//printf("start conv soft!\n");
    for(int chout=0;chout<cfg->CHout;chout++)
    {
        for(int hout=0;hout<cfg->Hout;hout++)
        {
            for(int wout=0;wout<cfg->Wout;wout++)
            {
            	int32_t tp1=0;
            	int32_t tp2;
//            	int32_t tmp;
//            	int32_t final_result;
                for(int chin=0;chin<cfg->CHin;chin++)
                {
                    for(int ky=0;ky<cfg->Ky;ky++)
                    {
                        for(int kx=0;kx<cfg->Kx;kx++)
                        {
                            int tp_data;
                            int tp_wt;
                            int hin;
                            int win;
                            hin=cfg->Sy*hout+ky-cfg->Py;
                            win=cfg->Sx*wout+kx-cfg->Px;
                            if( (hin<0) || (hin>=cfg->Hin) || (win<0) || (win>=cfg->Win) )//padding 0
                            {
                            	tp_data=0;
                            }
                            else
                            {   dat_in=addr_feature[chin*cfg->Hin*cfg->Win+hin*cfg->Win+win];
                            	tp_data=dat_in;
                            }
                                //tp_data=dat_in[chin*cfg->Hin*cfg->Win+hin*cfg->Win+win];
                                    wt=addr_weight[chout*cfg->CHin*cfg->Ky*cfg->Kx+chin*cfg->Ky*cfg->Kx+ky*cfg->Kx+kx];
                                    tp_wt=wt;
                                  //tp_wt=wt[chout*cfg->CHin*cfg->Ky*cfg->Kx+chin*cfg->Ky*cfg->Kx+ky*cfg->Kx+kx];
                            //tmp = tp1;
                            if(cfg->DAT_DW_L0==1){
                            	if(tp_data==tp_wt){tp1=tp1+1;}
                            	else{tp1=tp1-1;}
                            }
                            else{tp1=tp1 + (tp_data*tp_wt);}

                            //tp1=tp1 + (tp_data*tp_wt);
                            //if(chout==0 && hout==0 && wout==0)
                                //printf("tp1 = %d + %d x %d = %d \n",tmp,tp_data,tp_wt,tp1);
//                            if(tp_data>tp_wt)
//                            	tp1=tp1 - (tp_data-tp_wt);
//                            else
//                            	tp1=tp1 - (tp_wt-tp_data);
//                            printf("dat=%d, wt=%d, tp1=%d\n",tp_data,tp_wt,tp1);
                        }
                    }
                }
                // final_result=tp1;

                //dat_out[chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout]=final_result;
               // Xil_Out32( CONV_SOLFT + 4*(chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout), final_result);
               // if(chout==12 && hout==10 && wout==6)
              //    printf("-------------------final_result=%d dat_out[0]=%d \n", final_result,  Xil_In32( CONV_SOLFT + 4*(chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout)));

                if(shift_sign==0)//right shift and round
                {
                    tp2=tp1>>shift_value;
                    //if((shift_value!=0)&&(shift_sign!=0)&&(tp2!=(1<<(MAX_DW2+base_log2Tin+log2_KyKx+log2_other-1))-1))
                    //   tp2=tp2+(tp_feature_out[chout][hout][wout][out_truncate-1]);//$display("tp2=%0d",$signed(tp2));
                    if(cfg->DAT_DW_L1==1)
                        tp_sat=(tp2<0)?1:-1;
                    else if(cfg->DAT_DW_L1==2)
                    {
                        if(tp2<(-2)){tp_sat=-2;}
                        else if(tp2>1){tp_sat=1;}
                        else{tp_sat=tp2;}
                    }
                    else if(cfg->DAT_DW_L1==4)
                    {
                        if(tp2<(-8)){tp_sat=-8;}
                        else if(tp2>7){tp_sat=7;}
                        else{tp_sat=tp2;}
                    }
                    else if(cfg->DAT_DW_L1==8)
                    {
                        if(tp2<(-128)){tp_sat=-128;}
                        else if(tp2>127){tp_sat=127;}
                        else{tp_sat=tp2;}
                    }
                }
                else
                {
                    tp2=tp1<<shift_value;
                    tp_sat=tp2;
                }
                if(cfg->RELU_EN_L0==1){
                    if(tp_sat>=0){CONV_SOLFT[chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout]=tp_sat;}
                    else{CONV_SOLFT[chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout]=0;}
                }
                else{
                    CONV_SOLFT[chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout]=tp_sat;
                }
               
            }
        }
    }
}



void Rand_Gen(conv_cfg *cfg , int8_t *feature_rand, int len_rand_feature, int8_t *weight_rand, int len_rand_weight, bool low_value_test){

	printf("len_rand_feature= %d\n",len_rand_feature);
	printf("len_rand_weight= %d\n",len_rand_weight);
//random date generate
    if(cfg->DAT_DW_L0==1){
	    for(int i=0; i<len_rand_feature; i++)
        {
	    	feature_rand[i] = rand()%2;
            //printf("feature_rand[%d] = %d \n", i, feature_rand[i]);
	    }
	    for(int i=0; i<len_rand_weight; i++)
        {
	    	weight_rand[i] = rand()%2;
            //printf("weight_rand[%d] = %d \n", i, weight_rand[i]);
	    }
    }else if(low_value_test){
	    for(int i=0; i<len_rand_feature; i++)
        {
	    	feature_rand[i] = rand()%10-5;//((2<<(cfg->DAT_DW_L0-1))-1)-(2<<(cfg->DAT_DW_L0-2));
            //printf("feature_rand[%d] = %d \n", i, feature_rand[i]);
	    }
	    for(int i=0; i<len_rand_weight; i++)
        {
	    	weight_rand[i] = rand()%10-5;//((2<<(cfg->DAT_DW_L0-1))-1)-(2<<(cfg->DAT_DW_L0-2));
            //printf("weight_rand[%d] = %d \n", i, weight_rand[i]);
	    }
    }
    else 
    {
	    for(int i=0; i<len_rand_feature; i++)
        {
	    	feature_rand[i] = rand()%((2<<(cfg->DAT_DW_L0-1))-1)-(2<<(cfg->DAT_DW_L0-2));
            //printf("feature_rand[%d] = %d \n", i, feature_rand[i]);
	    }
	    for(int i=0; i<len_rand_weight; i++)
        {
	    	weight_rand[i] = rand()%((2<<(cfg->DAT_DW_L0-1))-1)-(2<<(cfg->DAT_DW_L0-2));
            //printf("weight_rand[%d] = %d \n", i, weight_rand[i]);
	    }
    }

}

void Read_data(conv_cfg *cfg , char *feature_file, int8_t *read_feature, int len_read_feature, char *weight_file, int8_t *read_weight, int len_read_weight, uint32_t in_scale, uint32_t wt_scale){

	printf("len_read_feature= %d\n",len_read_feature);
	printf("len_read_weight= %d\n",len_read_weight);
	FILE *fp0;
    int read_feature_tmp;
    int read_weight_tmp;
	fp0=fopen(feature_file,"r");
	if (fp0==NULL)
	{
		printf("Can't open feature file\n");
		exit(0);
	}
	for(int i=0;i<len_read_feature;i++)
	{
		if(fscanf(fp0,"%d",&read_feature_tmp))
        {
            if(cfg->DAT_DW_L0==1)
            {
                read_feature[i]=(read_feature_tmp+1)/2;
            }else{
                read_feature[i]=read_feature_tmp >> wt_scale;
            }
            //printf("read_feature[%d] = %d \n", i, read_feature[i]);
        }
        else
        {
            printf("fscanf read_feature_tmp error !!!");
        }

		// printf("Software_Output[%d] = %d \n", i, Software_Output[i]);
	}
	fclose(fp0);

    FILE *fp1;
	fp1=fopen(weight_file,"r");
	if (fp1==NULL)
	{
		printf("Can't open weight file\n");
		exit(0);
	}
	for(int i=0;i<len_read_weight;i++)
	{
		if(fscanf(fp1,"%d",&read_weight_tmp))
        {
            if(cfg->DAT_DW_L0==1)
            {
                read_weight[i]=(read_weight_tmp+1)/2;
            }else{
                read_weight[i]=read_weight_tmp >> in_scale;
            }
            //printf("read_weight[%d] = %d \n", i, read_weight[i]);
        }
        else
        {
            printf("fscanf read_weight_tmp error !!!");
        }
        
		// printf("Software_Output[%d] = %d \n", i, Software_Output[i]);
	}
	fclose(fp1);

}

void Read_bin_data(conv_cfg *cfg , char *feature_file, int8_t *read_feature, int len_read_feature, char *weight_file, int8_t *read_weight, int len_read_weight, uint32_t in_scale, uint32_t wt_scale){

	printf("len_read_feature= %d\n",len_read_feature);
	printf("len_read_weight= %d\n",len_read_weight);
	FILE *fp0;
    int read_feature_tmp;
    int read_weight_tmp;

	fp0=fopen(feature_file,"r");
    if (fp0==NULL)
	{
		printf("Can't open feature file\n");
		exit(0);
	}

    size_t rd_feature_size=fread(read_feature, 1, len_read_feature, fp0);

    if(rd_feature_size!=len_read_feature)
    {
        printf("fread read_feature error !!!");
    }
    else
    {
            if(cfg->DAT_DW_L0==1)
            {
                for(int i=0;i<len_read_feature;i++)
	            {
                    read_feature[i]=(read_feature[i]+1)/2;        
                //  printf("read_feature[%d] = %d \n", i, read_feature[i]);
	            }
                
            }
            else if(in_scale !=0){
                for(int i=0;i<len_read_feature;i++)
	            {
                    read_feature[i]=read_feature[i] >> in_scale;  
                //  printf("read_feature[%d] = %d \n", i, read_feature[i]);
	            }
                
            }
	    // for(int i=0;i<len_read_feature;i++)
	    // {        
        //     printf("read_feature[%d] = %d \n", i, read_feature[i]);
	    // }
    }
	fclose(fp0);

    FILE *fp1;
	fp1=fopen(weight_file,"r");
	if (fp1==NULL)
	{
		printf("Can't open weight file\n");
		exit(0);
	}

    size_t rd_weight_size=fread(read_weight, 1, len_read_weight, fp1);

    if(rd_weight_size!=len_read_weight)
    {
        printf("fread read_weight error !!!");
    }
    else
    {
            if(cfg->DAT_DW_L0==1)
            {
                for(int i=0;i<len_read_weight;i++)
	            {
                    read_weight[i]=(read_weight[i]+1)/2;        
                //  printf("read_feature[%d] = %d \n", i, read_feature[i]);
	            }
                
            }
            else if(wt_scale !=0){
                for(int i=0;i<len_read_weight;i++)
	            {
                    read_weight[i]=read_weight[i] >> wt_scale;  
                //  printf("read_feature[%d] = %d \n", i, read_feature[i]);
	            }
                
            }
    }
	fclose(fp1);

}

void config_on_chip_clock(conv_cfg *cfg, uint32_t CFG_REG_ADDR){
    //config the clock
	uint8_t clk_en = 1;
	uint8_t div_en = 1;
	// uint8_t freqset= 31;
	// uint8_t divset = 7;
	// uint8_t divset_pad =7;
    //first config all zero
	uint32_t inv_osc=(1<<16)+(1<<13)+0;
    Xil_Out32(CFG_REG_ADDR, inv_osc);
    //then config the real parameter
    printf("reset inv_osc =%x \n",inv_osc);
	inv_osc = (1<<16)+(1<<13) + clk_en + (div_en<<1) + (cfg->freqset<<2) + (cfg->divset <<7) + (cfg->divset_pad <<10);
    Xil_Out32(CFG_REG_ADDR, inv_osc);
    printf("cfg inv_osc =%x \n",inv_osc);
    Xil_Out32(CFG_REG_ADDR, 0);
    //After config the clock, the accelerator should be reset aggain.
}

void config_on_chip_register(conv_cfg *cfg, uint32_t CFG_REG_ADDR, uint32_t *cmd_cfg){
    cmd_cfg[0]=(1<<16)+(2<<13)+(2<<10)+(0<<8)+(cfg->Tin_factor<<3)+cfg->out_data_width; //24 //{spares_mode,hp_clk_sel,Tin_factor[4:0],out_data_width[2:0]}[9:0]
    cmd_cfg[1]=(1<<16)+(3<<13)+(cfg->OUT_Shift<<6)+(cfg->RELU_EN_L0<<5)+ cfg->practical_log2_cycle;
    cmd_cfg[2]=(1<<16)+(4<<13)+cfg->acc_times_minus1;
    cmd_cfg[3]=(1<<16)+(5<<13)+cfg->pixel_group_num_minus1;
    cmd_cfg[4]=(1<<16)+(6<<13)+cfg->input_data_num-1;
    cmd_cfg[5]=(1<<16)+(7<<13)+cfg->output_data_num-1;
    cmd_cfg[6]=0; //Why cmd_cfg[6]=0 make error?
    //Xil_Out32(CFG_REG_ADDR, (1<<16)+(3<<13)+(2<<10)+(1<<8));
    //printf("&cmd_cfg[6] = %x \n", &cmd_cfg[6]);//config finsh
    for(int i=0; i<6; i++)
    {
    	//printf("&cmd_cfg[%d] = %x \n", i, &cmd_cfg[i]);//config finsh
    	//printf("config[%d]: %x to addr: %x\n", i, cmd_cfg[i],CFG_REG_ADDR);
//        for(int j=0; j<10; j++)
//    	{
        	Xil_Out32(CFG_REG_ADDR, cmd_cfg[i]);
            sleep(0.001);
        	//printf("cmd_cfg[%d]=%x \n",i,cmd_cfg[i]);
//    	}
    }
    Xil_Out32(CFG_REG_ADDR,0); //after config , reset to 0;
}

void check_on_chip_register(conv_cfg *cfg, uint32_t CFG_REG_ADDR, uint32_t GET_CFG_REG_ADDR, uint32_t MODE_SEL_ADDR, uint8_t *addr_reg){

	printf("Stare check on-chip register finish: \n");
    //PS: read address of register start from 1, not 0;
    addr_reg[ 1]=1 + (1<<1) + (cfg->freqset<<2) + (cfg->divset <<7);
    addr_reg[ 2]=(cfg->divset >>1)+(cfg->divset_pad <<2);
	addr_reg[ 3]=(cfg->Tin_factor<<3)+cfg->out_data_width;
	addr_reg[ 4]=0;
	addr_reg[ 5]=(cfg->RELU_EN_L0<<5)+ cfg->practical_log2_cycle;
	addr_reg[ 6]=(cfg->OUT_Shift<<6);
	addr_reg[ 7]=cfg->acc_times_minus1;
	addr_reg[ 8]=cfg->pixel_group_num_minus1;
	addr_reg[ 9]=cfg->input_data_num-1;
	addr_reg[10]=(cfg->input_data_num-1)>>8;
	addr_reg[11]=cfg->output_data_num-1;
	addr_reg[12]=(cfg->output_data_num-1)>>8;

    //Before Read register, model sel should be set 1;
    Xil_Out32(MODE_SEL_ADDR,1);
	printf("model_sel = %d \n", Xil_In32(MODE_SEL_ADDR));

    uint32_t get_reg_dat=0;
    bool test_result;
    for(int i=1; i<=12; i++)
    {
    	//printf("set CFG_REG_ADDR: %x \n", CFG_REG_ADDR);
    	Xil_Out32(CFG_REG_ADDR, (1<<16)+(1<<5)+i);
    	//printf("set GET_CFG_REG_ADDR: %x \n", GET_CFG_REG_ADDR);
        get_reg_dat=Xil_In32(GET_CFG_REG_ADDR);
        //printf("set GET_CFG_REG_ADDR \n");
        test_result=(get_reg_dat==addr_reg[i]);
    	printf("%d: read_cfg_reg[%02d]= %02x; right_cfg_reg[%02d] = %02x \n", test_result, i, get_reg_dat, i, addr_reg[i]);
    }
    Xil_Out32(CFG_REG_ADDR, (1<<16)+0);
    Xil_Out32(CFG_REG_ADDR, 0);
    Xil_Out32(MODE_SEL_ADDR,0);//After Read register, model sel should be set 0;
}

uint8_t read_on_chip_register(uint32_t CFG_REG_ADDR, uint32_t GET_CFG_REG_ADDR, uint32_t MODE_SEL_ADDR, uint8_t addr_reg){

   //Before Read register, model sel should be set 1;
    Xil_Out32(MODE_SEL_ADDR,1);
    uint8_t get_reg_dat=0;
    Xil_Out32(CFG_REG_ADDR, (1<<16)+(1<<5)+0);
    get_reg_dat=Xil_In32(GET_CFG_REG_ADDR);
    //printf("read cfg_reg[%d]= %d \n", 0, get_reg_dat);
    Xil_Out32(CFG_REG_ADDR, (1<<16)+(1<<5)+addr_reg);
    get_reg_dat=Xil_In32(GET_CFG_REG_ADDR);
    //printf("read cfg_reg[%d]= %d \n", addr_reg, get_reg_dat);
    Xil_Out32(CFG_REG_ADDR, (1<<16)+0);
    Xil_Out32(CFG_REG_ADDR, 0);
    Xil_Out32(MODE_SEL_ADDR,0);//After Read register, model sel should be set 0;
    return get_reg_dat;
}

void read_all_on_chip_register(uint32_t CFG_REG_ADDR, uint32_t GET_CFG_REG_ADDR, uint32_t MODE_SEL_ADDR){
    for(uint8_t i=0; i<32; i++)
    {
    	read_on_chip_register(CFG_REG_ADDR, GET_CFG_REG_ADDR, MODE_SEL_ADDR, i);
    }
}
void start_vsc(uint32_t START_VSC_ADDR){
    //printf("Start VSC \n");
    Xil_Out32(START_VSC_ADDR,1);
    //printf("Finish Start VSC \n");
    //usleep(1000000);
    sleep(0.001);
    Xil_Out32(START_VSC_ADDR,0);

//start send data
    sleep(0.001);
    //printf("Start send data \n");
    Xil_Out32(START_VSC_ADDR,2);
    //printf("Finish Start send data \n");
	//Xil_Out32(START_VSC,0); //FPGA atto set to 0
}

bool waiting_finish(uint32_t GET_OUT_VALID_ADDR, uint32_t GET_DONE_ADDR){
	uint32_t out_valid=0;
	 //printf("Waiting VSC out valid...... \n");
     do
     {
       out_valid = Xil_In32(GET_OUT_VALID_ADDR);
       sleep(0.001);
       //printf("chip state= %x \n", read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 17));
     } while ( out_valid != 1 );
    //printf("VSC out valid !!! \n");

    //printf("Waiting VSC out valid=%d \n",Xil_In32(GET_OUT_VALID_ADDR));

	uint32_t finish=0;
	//printf("Waiting VSC finish...... \n");
    do
    {
       finish = Xil_In32(GET_DONE_ADDR);
       sleep(0.001);
       //printf("chip state= %x \n", read_on_chip_register(CFG_REG, GET_CFG_REG, MODE_SEL, 17));
    } while ( finish != 1 );

 	//printf("VSC finish !!! \n");

    return (finish==1);
}


void Write_feature(conv_cfg *cfg, uint32_t *feature, uint32_t write_addr){
//write dat
	    uint32_t dat_len = cfg->input_data_num * 256/16;
	    uint16_t reshape_dt;
	    uint16_t wr_dt_addr;
	    uint32_t wr_dt;
			if(cfg->DAT_DW_L0==8){
				int cnt = 0;
				int cnt_addr = 0;
				for(int i=0; i< dat_len ;i++)
				{
					wr_dt_addr=i;
					if(cnt < 8) {
                        reshape_dt=feature[cnt_addr];
						// reshape_dt=Xil_In16(BASE_DAT+cnt_addr*2);
						//printf("dt[%d]=%x \n", cnt_addr, reshape_dt);
						//reshape_dt=1;
						wr_dt=(wr_dt_addr<<16)+reshape_dt;
                        //wr_dt=(wr_dt_addr<<16)+0;
						cnt_addr=cnt_addr+1;
					}
					else {
						wr_dt=(wr_dt_addr<<16)+0;//+0
					}
					//printf("write feature[%d] = %d , cnt_addr= %d , reshape_dt = %d\n", wr_dt>>16, (wr_dt<<16)>>16,cnt_addr, reshape_dt);
					Xil_Out32(write_addr, wr_dt);
					if(cnt == 15){
						cnt = 0; }
					else {
						cnt = cnt +1 ;
					}
			    }
			}
			else
			{
				for(int i=0; i< dat_len ;i++) {
					wr_dt_addr=i;
                    reshape_dt=feature[i];
					//reshape_dt=Xil_In16(BASE_DAT+i*2);
//					if(i%100==0){
//					printf("reshape_dt[%d] = %x \n", i, reshape_dt);
//					}
					//reshape_dt=1;
                    
					wr_dt=(wr_dt_addr<<16)+reshape_dt;
                    //printf("write feature[%d] = %d , cnt_addr= %d , reshape_dt = %d\n", wr_dt>>16, (wr_dt<<16)>>16, i, reshape_dt);
					Xil_Out32(write_addr, wr_dt);
					}
		    }

}

void Write_weight(conv_cfg *cfg, uint32_t *weight, uint32_t write_addr){
//write wt
	    uint32_t wt_len =cfg->input_data_num * 256/16;
	    uint16_t reshape_wt;
	    uint16_t wr_wt_addr;
	    uint32_t wr_wt;

		if(cfg->DAT_DW_L0==8)
		{
			int cnt = 0;
			int cnt_addr = 0;
			for(int i=0; i< wt_len ;i++)
			   {
				wr_wt_addr=i;
				    if(cnt < 8)
				    {
                       reshape_wt=weight[cnt_addr];
					   //reshape_wt=Xil_In16(BASE_WT+cnt_addr*2);
					   //printf("wt[%d]=%x \n", cnt_addr, reshape_wt);
					   //reshape_wt=1;
					   wr_wt=(wr_wt_addr<<16)+reshape_wt;
                       //wr_wt=(wr_wt_addr<<16)+1;
					   cnt_addr=cnt_addr+1;
				     }
				    else
				    {
					    wr_wt=(wr_wt_addr<<16)+0;//+0;
				    }
				    //printf("write weight[%d] = %d \n", wr_wt>>16, (wr_wt<<16)>>16);
				    Xil_Out32(write_addr, wr_wt);
                   if(cnt == 15){ cnt = 0;}
				    else {cnt = cnt +1 ;}
		    }
		}
		else
		{
			for(int i=0; i< wt_len ;i++) {
				//reshape_wt=Xil_In16(BASE_WT+i*2);
				wr_wt_addr=i;
                reshape_wt=weight[i];
				// reshape_wt=Xil_In16(BASE_WT+i*2);
//				if(i%100==0){
//				printf("reshape_wt[%d] = %x \n", i, reshape_wt);
//				}
				wr_wt=(wr_wt_addr<<16)+reshape_wt;
                //printf("write weight[%d] = %d \n", wr_wt>>16, (wr_wt<<16)>>16);

				Xil_Out32(write_addr, wr_wt);
				}
	    }

}

void read_chip_out_result(conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR)
{
	printf("READ_ODAT_ADDR: %x \n", READ_ODAT_ADDR);
	int8_t tmp_Out;
    uint32_t rd_addr;
    for(int chout=0;chout<cfg->slice_of_CHout_L0;chout++)
        for(int hout=0;hout<cfg->Hout;hout++)
            for(int wout=0;wout<cfg->Wout;wout++)
                 for(int tout=0;tout<Tout;tout++)
                 {
          	       rd_addr=chout*cfg->Hout*cfg->Wout*Tout + hout*cfg->Wout*Tout + wout*Tout+tout;
	               Xil_Out32(READ_ADDR_PARE, rd_addr);
	               //printf("READ_ODAT_ADDR: %x", READ_ODAT_ADDR);
			       tmp_Out=Xil_In32(READ_ODAT_ADDR);
			       //printf("chip_out[%d] = %d \n", rd_addr, tmp_Out);
			       //printf("out_tmp[%d]=%d \n",rd_addr,tmp_Out);
				   chip_out_result[ (chout*Tout+tout)*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout]=tmp_Out;
			       //Xil_Out8( DEST + (chout*Tout+tout)*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout, tmp_Out);
                     //dat_out_FPGA[chout*`Tout+tout][hout][wout]=u_DDR_Output_Feature.memory[chout*`Hout_L0*`Wout_L0 + hout*`Wout_L0 + wout][tout*`MAX_DW+:`MAX_DW];
                 }
}

void read_chip_out_result_div(conv_cfg *cfg_div, conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR, uint32_t test_i, uint32_t test_j)
{
	//printf("READ_ODAT_ADDR: %x \n", READ_ODAT_ADDR);
	int8_t tmp_Out;
    uint32_t rd_addr;
    for(int chout=0;chout<cfg_div->slice_of_CHout_L0;chout++)
        for(int hout=0;hout<cfg_div->Hout;hout++)
            for(int wout=0;wout<cfg_div->Wout;wout++)
                 for(int tout=0;tout<Tout;tout++)
                 {
          	       rd_addr=chout*cfg_div->Hout*cfg_div->Wout*Tout + hout*cfg_div->Wout*Tout + wout*Tout+tout;
	               Xil_Out32(READ_ADDR_PARE, rd_addr);
	               //printf("READ_ODAT_ADDR: %x", READ_ODAT_ADDR);
			       tmp_Out=Xil_In32(READ_ODAT_ADDR);
                   uint32_t ptr=(chout*Tout+tout+test_i*Tout)*cfg->Hout*cfg->Wout+hout*cfg_div->Wout+wout+test_j*Tout;
			       
			       //printf("out_tmp[%d]=%d \n",rd_addr,tmp_Out);
				   chip_out_result[ptr]=tmp_Out;
                   //printf("chip_out[%d] = %d, chip_out_result [%d]= %d\n", rd_addr, tmp_Out, ptr,chip_out_result[ptr]);
			       //Xil_Out8( DEST + (chout*Tout+tout)*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout, tmp_Out);
                     //dat_out_FPGA[chout*`Tout+tout][hout][wout]=u_DDR_Output_Feature.memory[chout*`Hout_L0*`Wout_L0 + hout*`Wout_L0 + wout][tout*`MAX_DW+:`MAX_DW];
                 }
}

void read_chip_out_result_div_faster(conv_cfg *cfg_div, conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR, uint32_t test_i, uint32_t test_j, uint32_t div_KxKyCin)
{
	//printf("READ_ODAT_ADDR: %x \n", READ_ODAT_ADDR);
	int8_t tmp_Out;
    uint32_t rd_addr;
    for(int chout=0;chout<cfg_div->slice_of_CHout_L0;chout++)
        for(int hout=0;hout<cfg_div->Hout;hout++)
            for(int wout=0;wout<cfg_div->Wout;wout++)
                 for(int tout=0;tout<Tout;tout++)
                 {
          	       rd_addr=chout*cfg_div->Hout*cfg_div->Wout*Tout + hout*cfg_div->Wout*Tout + wout*Tout+tout;
	               Xil_Out32(READ_ADDR_PARE, rd_addr);
	               //printf("READ_ODAT_ADDR: %x", READ_ODAT_ADDR);
			       tmp_Out=Xil_In32(READ_ODAT_ADDR);
                   uint32_t ptr=(chout*Tout+tout+test_i*Tout)*cfg->Hout*cfg->Wout+hout*cfg_div->Wout+wout+test_j*Tout*div_KxKyCin;
			       
			       //printf("out_tmp[%d]=%d \n",rd_addr,tmp_Out);
				   chip_out_result[ptr]=tmp_Out;
                   //printf("chip_out[%d] = %d, chip_out_result [%d]= %d\n", rd_addr, tmp_Out, ptr,chip_out_result[ptr]);
			       //Xil_Out8( DEST + (chout*Tout+tout)*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout, tmp_Out);
                     //dat_out_FPGA[chout*`Tout+tout][hout][wout]=u_DDR_Output_Feature.memory[chout*`Hout_L0*`Wout_L0 + hout*`Wout_L0 + wout][tout*`MAX_DW+:`MAX_DW];
                 }
}

void read_chip_out_result_div_linera(conv_cfg *cfg_div, conv_cfg *cfg, int8_t *chip_out_result, uint32_t READ_ADDR_PARE,  uint32_t READ_ODAT_ADDR, 
                                     uint32_t test_i, uint32_t test_j, uint32_t div_KxKyCin, uint32_t test_k)
{
	//printf("READ_ODAT_ADDR: %x \n", READ_ODAT_ADDR);
	int32_t tmp_Out;
    uint32_t rd_addr;
    for(int chout=0;chout<cfg_div->slice_of_CHout_L0;chout++)
        for(int hout=0;hout<cfg_div->Hout;hout++)
            for(int wout=0;wout<cfg_div->Wout;wout++)
                 for(int tout=0;tout<Tout;tout++)
                 {
          	       rd_addr=chout*cfg_div->Hout*cfg_div->Wout*Tout + hout*cfg_div->Wout*Tout + wout*Tout+tout;
	               Xil_Out32(READ_ADDR_PARE, rd_addr);
	               //printf("READ_ODAT_ADDR: %x", READ_ODAT_ADDR);
			       tmp_Out=Xil_In32(READ_ODAT_ADDR);
                   uint32_t ptr=(chout*Tout+tout+test_i*Tout)*cfg->Hout*cfg->Wout+hout*cfg_div->Wout+wout+test_j*Tout*div_KxKyCin;
			       
			       //printf("out_tmp[%d]=%d , chip_out_result[%d] = %d\n",rd_addr,tmp_Out, ptr, chip_out_result[ptr]);
                   if(test_k >0)
                   {
                        tmp_Out=tmp_Out+chip_out_result[ptr];
                   }
                     
                    if(tmp_Out>127){
                        tmp_Out = 127;
                    }
                    else if (tmp_Out<-128){
                        tmp_Out = -128;
                    }
                    if((chout*Tout+tout+test_i*Tout) < cfg->CHout)
                    {
                        chip_out_result[ptr]=tmp_Out;
                    }
                    else 
                    {
                        //printf("Error with out of range CHout = %d, but chout of ptr is %d \n", cfg->CHout , (chout*Tout+tout+test_i*Tout));
                    }
                    
                   
				   //printf("prt[%d][%d][%d] = %d , chout = %d, tout = %d, test_i = %d \n", (chout*Tout+tout+test_i*Tout), hout, wout+test_j*Tout*div_KxKyCin, ptr, chout, tout, test_i);
                   //printf("chip_out[%d] = %d, chip_out_result [%d]= %d\n", rd_addr, tmp_Out, ptr,chip_out_result[ptr]);
                   //printf("chip_out_result[%d][%d][%d] = %d\n", (chout*Tout+tout+test_i*Tout), hout, wout+test_j*Tout*div_KxKyCin, chip_out_result[ptr]);
			       //Xil_Out8( DEST + (chout*Tout+tout)*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout, tmp_Out);
                     //dat_out_FPGA[chout*`Tout+tout][hout][wout]=u_DDR_Output_Feature.memory[chout*`Hout_L0*`Wout_L0 + hout*`Wout_L0 + wout][tout*`MAX_DW+:`MAX_DW];
                 }
}

void compare_result(conv_cfg *cfg, char *d0_name, int8_t *chip_out_result, char *d1_name, int8_t *conv_solft, bool out_error, bool print_all)
{

    int8_t error_flg=0;
	int error_cnt=0;
	int8_t tmp_out_conv_solft;
    int8_t tmp_out_chip_out_result;
    for(int chout=0;chout<cfg->CHout;chout++)
    {
        for(int hout=0;hout<cfg->Hout;hout++)
        {
            for(int wout=0;wout<cfg->Wout;wout++)
            {
				//Xil_In8( CONV_SOLFT + (chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout));
            	tmp_out_conv_solft      =conv_solft[(chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout)];
				//Xil_In8( DEST + (chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout));
				tmp_out_chip_out_result =chip_out_result[(chout*cfg->Hout*cfg->Wout+hout*cfg->Wout+wout)];
				if(print_all)
				{
				printf("%s[%d][%d][%d] = %d  %s[%d][%d][%d] = %d \n",
			           d0_name, chout,hout,wout,tmp_out_chip_out_result,
					   d1_name, chout,hout,wout,tmp_out_conv_solft);
				}
            	if(tmp_out_conv_solft!=tmp_out_chip_out_result)
				{
					error_flg=1;
					error_cnt=error_cnt+1;
				    if(out_error)
			        {
			            printf("%s[%d][%d][%d] = %d  %s[%d][%d][%d] = %d \n",
					           d0_name, chout,hout,wout,tmp_out_chip_out_result,
							   d1_name, chout,hout,wout,tmp_out_conv_solft);
					}
				}
            }
        }
    }
	printf("%s vs %s compare result: error_cnt/Total_num =  %d / %d \n",
	          d0_name, d1_name, error_cnt,cfg->CHout*cfg->Hout*cfg->Wout);
    if(error_flg==1){
        printf("%s vs %s Mismatch \n", d0_name, d1_name);

    }else{
         printf("%s vs %s Result match \n", d0_name, d1_name);
    }

}


void RunConv_Reshape(conv_cfg *cfg, int8_t *addr_feature, int8_t *addr_weight,
                                    uint32_t *BASE_DAT, int32_t len_re_feature,
                                    uint32_t *BASE_WT, int32_t len_re_weight )
{
	int8_t in_scale=0;
	int8_t wt_scale=0;
	int8_t out_scale=cfg->OUT_Shift;

	int8_t feat_in;
	int8_t weight;
	uint16_t tp_dt, tp_wt;

	uint8_t tpp_dt, tpp_wt;
    int32_t h, w, row, col;
    uint32_t Chin_tmp=0;
    int32_t cnt_dat=0;
    int32_t cnt_wt=0;
    int32_t cnt_t=0;
    int32_t cnt_tt = 0;
    int32_t slice_output_pixels = ((cfg->Wout*cfg->Hout+Tout-1)/Tout);
    int32_t all_for=len_re_feature; //cfg->slice_of_CHout_L0*slice_output_pixels*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
    uint32_t prt_dt;
    printf("BASE_DAT = %p \n", BASE_DAT);
    printf("BASE_WT = %p \n", BASE_WT);
	//printf("%d=%dx%dx%dx%dx%dx%dx%d \n",all_for,cfg->slice_of_CHout_L0,slice_output_pixels,cfg->slice_of_CHin_L0, cfg->Ky,cfg->Kx,Tout, cfg->Tin_num/cfg->div_num);
    printf("cnt_dat/all_for_dat = %d/%d \n", cnt_t,all_for);
    //reshape feature
        for(int chout=0;chout<cfg->slice_of_CHout_L0;chout++) {
            for(int p=0;p<slice_output_pixels;p++) {
                for(int chin=0;chin<cfg->slice_of_CHin_L0;chin++) {
                    for(int ky=0;ky<cfg->Ky;ky++){
                        for(int kx=0;kx<cfg->Kx;kx++) {
                            for(int pp=0;pp<Tout;pp++) {
                                h=(p*Tout+pp) / cfg->Wout;
                                w=(p*Tout+pp) % cfg->Wout;
                                row=h*cfg->Sy - cfg->Py + ky;
                                col=w*cfg->Sx - cfg->Px + kx;
                                //for(int tout=0;tout<Tout;tout++) {
                                	for(int tin_div=0; tin_div < cfg->Tin_num/cfg->div_num; tin_div++)
                                    {
                                        tp_dt = 0;
                                        for(int tin=0;tin<cfg->div_num;tin++)
                                        {
                                        	tpp_dt =0;
                                            for (int dw=0; dw < cfg->dw_num; dw++)
                                            {
                                            	Chin_tmp=(chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw);
                                                if (row<0 || col<0 || row >= cfg->Hin || col >= cfg->Win ||  Chin_tmp>= cfg->CHin)
                                                {
                                                	tpp_dt = tpp_dt+ (0 << (cfg->dw * dw));
                                                }

                                                else
                                                {
                                                	      feat_in=addr_feature[in_index(cfg, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, row, col)];
									    			      if(cfg->DAT_DW_L0==1)
									    				  {
									    			    	  tpp_dt = tpp_dt+((feat_in&0x01)<< (cfg->dw * dw));
                                                	          //tpp_dt = tpp_dt+( (feat_in[cfg->in_index(cfg, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, row, col)]&0x01)<< (cfg->dw * dw));
                                                          }
									    				  else if(cfg->DAT_DW_L0==2)
                                                          {
                                                              tpp_dt = tpp_dt+((feat_in&0x03)<< (cfg->dw * dw));
                                                          }
                                                          else if(cfg->DAT_DW_L0==4)
                                                          {
                                                              tpp_dt = tpp_dt+((feat_in&0x0f)<< (cfg->dw * dw));
                                                          }
                                                          else if(cfg->DAT_DW_L0==8)
                                                          {
                                                              tpp_dt = tpp_dt+(feat_in << (cfg->dw * dw));
                                                          }
                                                }
                                            }
                                            tp_dt = tp_dt + (tpp_dt << tin*8);
                                        }
//                                    if(cnt_t%100==0){
//                                    					printf("tp_dt[%d] = %x \n", cnt_t, tp_dt);
//                                    					}
                                            prt_dt=cnt_dat*cfg->Tin_num/cfg->div_num+tin_div;
                                            // if(BASE_DAT[prt_dt]>BASE_DAT[all_for]|| (BASE_DAT[prt_dt])< (BASE_DAT)){
                                            //       printf("BASE_DAT[%d] out of rang ! &BASE_DAT[%d]=%p, &BASE_DAT[%d]=%p, &BASE_DAT=%p\n", prt_dt, prt_dt, &BASE_DAT[prt_dt],all_for, &BASE_DAT[all_for], BASE_DAT);
                                            // }else{}
                                            BASE_DAT[prt_dt]=tp_dt;
                                        // if(&BASE_DAT[cnt_dat*cfg->Tin_num/cfg->div_num+tin_div]>&BASE_DAT[all_for]|| (&BASE_DAT[cnt_dat*cfg->Tin_num/cfg->div_num+tin_div])< (&BASE_DAT)){
                                        //     printf("BASE_DAT[%d] out of rang !\n", cnt_dat*cfg->Tin_num/cfg->div_num+tin_div);
                                        // }else{BASE_DAT[cnt_dat*cfg->Tin_num/cfg->div_num+tin_div]=tp_dt;}
                                        
                                        //printf("BASE_DAT[%d] = %d tp_dt= %d \n",prt_dt,BASE_DAT[prt_dt],tp_dt);
                                        //printf("BASE_DAT[%d]=%d.......................... \n", 593, BASE_DAT[593]);
                                        //ASE_DAT[cnt_dat*cfg->Tin_num+tin_div*cfg->div_num]=tp_dt;
                                        cnt_t=cnt_t+1;
                                	}
                            //}
                                cnt_dat += 1;
                            }
                            //cnt_wt += Tout;
                        }
                    }
                }
                //printf("cnt_da/all_for_dat = %d/%d \n", cnt_t,all_for);
            }
        }
        printf("cnt_da/all_for_dat = %d/%d \n", cnt_t,all_for);
        all_for=len_re_weight;//cfg->slice_of_CHout_L0*slice_output_pixels*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;
        printf("cnt_wt/all_for_wt = %d/%d \n", cnt_tt,all_for);
        //reshape weight
        uint32_t prt_wt=0;
            for(int chout=0;chout<cfg->slice_of_CHout_L0;chout++) {
                for(int p=0;p<slice_output_pixels;p++) {
                    for(int chin=0;chin<cfg->slice_of_CHin_L0;chin++) {
                        for(int ky=0;ky<cfg->Ky;ky++){
                            for(int kx=0;kx<cfg->Kx;kx++) {
                                //for(int pp=0;pp<Tout;pp++) {
//                                    h=(p*Tout+pp) / cfg->Wout;
//                                    w=(p*Tout+pp) % cfg->Wout;
//                                    row=h*cfg->Sy - cfg->Py + ky;
//                                    col=w*cfg->Sx - cfg->Px + kx;
                                    for(int tout=0;tout<Tout;tout++)
                                    {
                                    	for(int tin_div=0; tin_div < cfg->Tin_num/cfg->div_num; tin_div++)
                                        {
                                            tp_wt = 0;
                                            for(int tin=0;tin<cfg->div_num;tin++)
                                            {
                                            	tpp_wt =0;
                                                for (int dw=0; dw < cfg->dw_num; dw++)
                                                {
                                                	Chin_tmp=(chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw);
                                                    if (Chin_tmp>= cfg->CHin)
                                                    {
                                                	  tpp_wt += 0<< (cfg->dw * dw);
                                                    }
                                                    else
                                                    {
                                                	      weight=addr_weight[wt_index(cfg, chout*Tout+tout, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, ky, kx)];
										    		      if(cfg->DAT_DW_L0==1)
										    			  {
										    		    	  tpp_wt += (weight&0x01)<< (cfg->dw * dw);
                                                	           //tpp_wt += (weight[cfg->wt_index(cfg, chout*Tout+tout, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, ky, kx)]&0x01)<< (cfg->dw * dw);
                                                          }
										    			  else if(cfg->DAT_DW_L0==2)
                                                          {
                                                               tpp_wt += (weight&0x03)<< (cfg->dw * dw);
                                                          }
                                                          else if(cfg->DAT_DW_L0==4)
                                                          {
                                                                tpp_wt += (weight&0x0f)<< (cfg->dw * dw);
                                                          }
                                                          else if(cfg->DAT_DW_L0==8)
                                                          {
                                                                tpp_wt += weight << (cfg->dw * dw);
                                                          }
                                                    }

                                                }
                                                tp_wt = tp_wt+ (tpp_wt<<(tin*8));
                                             }
//                                            if(cnt_tt%100==0){
//                                                                                					printf("tp_wt[%d] = %x \n", cnt_tt, tp_wt);
//                                                                                					}
                                            prt_wt=(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div;
                                            // if(&BASE_WT[prt_wt]>&BASE_WT[all_for]|| (&BASE_WT[prt_wt])< (&BASE_WT)){
                                            //       printf("BASE_WT[%d] out of rang ! &BASE_WT[%d]=%p, &BASE_WT[%d]=%p, &BASE_WT=%p\n", prt_wt, prt_wt, &BASE_WT[prt_wt],all_for, &BASE_WT[all_for], &BASE_WT);
                                            // }else{BASE_WT[prt_wt]=tp_wt;}
                                            BASE_WT[prt_wt]=tp_wt;
                                            //printf("cnt=%d  ", cnt_tt);
                                            //test_malloc();

                                            //printf("BASE_WT[%d] = %d, &BASE_WT[%d]= %x\n",(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div,BASE_WT[(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div], (cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div,&BASE_WT[(cnt_wt+tout)*cfg->Tin_num/cfg->div_num+tin_div]);
                                            //printf("BASE_DAT[%d]=%d.......................... &BASE_DAT[%d]= %x\n", 593, BASE_DAT[593], 593, &BASE_DAT[593]);
                                            //BASE_WT[(cnt_wt+tout)*cfg->Tin_num+tin_div*cfg->div_num]=tp_wt;
                                            cnt_tt=cnt_tt+1;
                                    	}
                                    }
                                    //cnt_dat += 1;
                                //}
                                cnt_wt += Tout;

                            }
                        }
                    }
                    //printf("cnt_wt/all_for_wt = %d/%d \n", cnt_tt,all_for);
                }
            }
            printf("cnt_wt/all_for_wt = %d/%d \n", cnt_tt,all_for);


}

void test_malloc(){
	int8_t *test = (int8_t *)malloc(1);
		if (test == NULL){
		    printf("test malloc with len : %d error!\n",1);
		}else{printf("test malloc with len : %d successful!\n",1);}
};
void Reshape_conv_soft(conv_cfg *cfg, int8_t *addr_feature,  int8_t *addr_weight, int8_t *addr_re_conv) {
	int8_t in_scale=0;
	int8_t wt_scale=0;
	int8_t out_scale=0;
    int16_t shift_value;
    int8_t shift_sign;
    //int16_t out_truncate;
    shift_sign=0;//right shift
    shift_value=(in_scale+wt_scale)-out_scale;
    if(shift_value<0){
        shift_value=out_scale-(in_scale+wt_scale);
        shift_sign=1; //left shift
    }

   // out_truncate={shift_sign,shift_value}; //$display("out_truncate=%b",out_truncate);
int32_t tp2;
int8_t tp_sat;

	int32_t tp_dt, tp_wt, tp;
	int8_t tpp_dt, tpp_wt;
    int32_t h, w, row, col;

    int32_t cnt_dat=0;
    int32_t cnt_wt=0;
    int32_t cnt_t=0;
    int32_t cnt_tt = 0;
    int32_t CHin_tmp=0;
    int32_t slice_output_pixels = ((cfg->Wout*cfg->Hout+Tout-1)/Tout);
    //printf("slice output pixel = %d\n", slice_output_pixels);

    uint32_t div_KxKyTout = WT_SRAM_DEPTH/(cfg->Ky*cfg->Kx*Tout);
    uint32_t div_slice_of_CHin_L0 = (cfg->slice_of_CHin_L0+div_KxKyTout-1)/div_KxKyTout;
    uint32_t div_KxKyCin = WT_SRAM_DEPTH/(cfg->Ky*cfg->Kx*Tout*cfg->slice_of_CHin_L0);
    uint32_t div_out_pixels = (slice_output_pixels+div_KxKyCin-1)/div_KxKyCin;


    int32_t tp_sum[Tout][Tout];
    int32_t tp_feature_out[cfg->CHout][cfg->Hout][cfg->Wout];
    for(int i=0;i<Tout;i++)
        for(int j=0;j<Tout;j++)
            tp_sum[i][j]=0;

    int32_t all_for=cfg->slice_of_CHout_L0*slice_output_pixels*cfg->slice_of_CHin_L0*cfg->Ky*cfg->Kx*Tout* cfg->Tin_num/cfg->div_num;

        for(int chout=0;chout<cfg->slice_of_CHout_L0;chout++)
        {
            //for(int p=0;p<slice_output_pixels;p++) {
            //for(int p=0;p<slice_output_pixels;p++) {
             for(int div_p=0;div_p<div_out_pixels;div_p++) {

                for(int div_c0=0;div_c0<div_KxKyCin;div_c0++) {
                    int p=div_p*div_KxKyCin+div_c0;

                for(int chin=0;chin<cfg->slice_of_CHin_L0;chin++) {
                    for(int ky=0;ky<cfg->Ky;ky++){
                        for(int kx=0;kx<cfg->Kx;kx++) {
                            for(int pp=0;pp<Tout;pp++) {
                                h=(p*Tout+pp) / cfg->Wout;
                                w=(p*Tout+pp) % cfg->Wout;
                                row=h*cfg->Sy - cfg->Py + ky;
                                col=w*cfg->Sx - cfg->Px + kx;
                                for(int tout=0;tout<Tout;tout++) {
                                	for(int tin_div=0; tin_div < cfg->Tin_num/cfg->div_num; tin_div++){
                                        tp_dt = 0;
                                        tp_wt = 0;
                                    for(int tin=0;tin<cfg->div_num;tin++) {
                                    	tpp_dt =0;
                                    	tpp_wt =0;
                                        for (int dw=0; dw < cfg->dw_num; dw++)
                                        {

                                        	CHin_tmp=(chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw);

                                            if (row<0 || col<0 || row >= cfg->Hin || col >= cfg->Win ||  CHin_tmp>= cfg->CHin){
                                            	tpp_dt = 0;
                                            	//printf("tpp_dt_0 = %d\n", tpp_dt);

                                            }

                                            else {

                                            	tpp_dt=addr_feature[in_index(cfg, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, row, col)];
                                            	//tpp_dt = feat_in[cfg->in_index(cfg, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, row, col)];
                                            	//printf("tpp_dt_none_0 = %d\n", tpp_dt);

                                            }


                                            tpp_wt =addr_weight[wt_index(cfg, chout*Tout+tout, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, ky, kx)];
                                            //tpp_wt = weight[cfg->wt_index(cfg, chout*Tout+tout, chin*cfg->Tin_l0+(tin+tin_div*cfg->div_num)*cfg->dw_num+dw, ky, kx)];

                                            if(cfg->DAT_DW_L0==1){
                                            	if(CHin_tmp >= cfg->CHin){
                                            		tp_sum[pp][tout]=tp_sum[pp][tout];
                                            	}else{
                                            	    if(tpp_dt==tpp_wt){tp_sum[pp][tout]=tp_sum[pp][tout]+1;}
                                            	    else{tp_sum[pp][tout]=tp_sum[pp][tout]-1;}
                                            	}
                                            }
                                            else{tp_sum[pp][tout]= tp_sum[pp][tout]+tpp_dt*tpp_wt;}
                                        }
                                    }
                                	}
                                }
                               // cnt_dat += 1;
                            }
                            //cnt_wt += Tout;
                        }
                    }
                }    //chinend
                for (int pp=0;pp<Tout;pp++)
                {    if (p*Tout+pp<(cfg->Hout*cfg->Wout))
                    {
                        h=(p*Tout+pp) / cfg->Wout;
                        w=(p*Tout+pp) % cfg->Wout;
                        for (int tout=0;tout<Tout;tout++)
                        {
                            tp=tp_sum[pp][tout];
                            tp_sum[pp][tout]=0;
                            if ((chout*Tout+tout)<cfg->CHout)
                            {
                                if(shift_sign==0)//right shift and round
                                {
                                    tp2=tp>>shift_value;
                                    //if((shift_value!=0)&&(shift_sign!=0)&&(tp2!=(1<<(MAX_DW2+base_log2Tin+log2_KyKx+log2_other-1))-1))
                                    //   tp2=tp2+(tp_feature_out[chout][hout][wout][out_truncate-1]);//$display("tp2=%0d",$signed(tp2));
                                    if(cfg->DAT_DW_L1==1)
                                        tp_sat=(tp2<0)?1:-1;
                                    else if(cfg->DAT_DW_L1==2)
                                    {
                                       if(tp2<(-2)){tp_sat=-2;}
                                       else if(tp2>1){tp_sat=1;}
                                       else{tp_sat=tp2;}
                                    }
                                    else if(cfg->DAT_DW_L1==4)
                                    {
                                       if(tp2<(-8)){tp_sat=-8;}
                                       else if(tp2>7){tp_sat=7;}
                                       else{tp_sat=tp2;}
                                    }
                                    else if(cfg->DAT_DW_L1==8)
                                    {
                                       if(tp2<(-128)){tp_sat=-128;}
                                       else if(tp2>127){tp_sat=127;}
                                       else{tp_sat=tp2;}
                                    }
                                }
                                else
                                {
                                    tp2=tp<<shift_value;
                                    tp_sat=tp2;
                                }

                                 if(cfg->RELU_EN_L0==1 && tp_sat<0){
                                      addr_re_conv[((chout*Tout+tout)*cfg->Hout*cfg->Wout+h*cfg->Wout+w)]= 0;
                                 }
                                 else{
                                     addr_re_conv[((chout*Tout+tout)*cfg->Hout*cfg->Wout+h*cfg->Wout+w)]= tp_sat;
                                 }                                
                                
                                //printf("addr_re_conv[%d]= %d \n", ((chout*Tout+tout)*cfg->Hout*cfg->Wout+h*cfg->Wout+w), addr_re_conv[((chout*Tout+tout)*cfg->Hout*cfg->Wout+h*cfg->Wout+w)]);

                            	//Xil_Out32( CONV_SOLFT2 + 4*((chout*Tout+tout)*cfg->Hout*cfg->Wout+h*cfg->Wout+w), tp);
                                //tp_feature_out[chout*Tout+tout][h][w]=tp;//$display("feature_out[%0d][%0d][%0d]=%0d",chout*`Tout+tout,h,w,$signed(tp));
                            }
                        }
                    }
                }
            }//pp_end
            }
        }

}

