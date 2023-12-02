.PHONY:clean all

LIB_SRC= ./lib/basic.c  \
	 ./lib/basic.h \
	 ./lib/cnn_defines.h \
	 ./lib/rand_structure.h

LIB_VGG= ./lib/basic.c  \
	 ./lib/basic.h \
	 ./lib/cnn_defines.h \
	 ./lib/model_structure.h

Compile_CMD= 	-O3

all:vgg_test_linear
 
clean:
	@rm -f test_rand hello

rand_test:
	gcc test_rand.c ${LIB_SRC} -I ./lib/ -I ./ ${Compile_CMD} -o test_rand

vgg_test:
	gcc test_vgg.c ${LIB_VGG} -I ./lib/ -I ./ ${Compile_CMD} -o test_vgg

vgg_test_faster:
	gcc test_vgg_faster.c ${LIB_VGG} -I ./lib/ -I ./ ${Compile_CMD} -o test_vgg_faster

vgg_test_linear:
	gcc test_vgg_linear.c ${LIB_VGG} -I ./lib/ -I ./ ${Compile_CMD} -o test_vgg_linear

clk_test:
	gcc test_clk.c ${LIB_SRC} -I ./lib/ -I ./ ${Compile_CMD} -o test_clk

test:
	gcc test.c ${LIB_SRC} -I ./lib/ -I ./ ${Compile_CMD} -o test

