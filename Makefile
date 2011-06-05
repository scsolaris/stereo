NVCCFLAGS 	+= -I/home/wuhao/NVIDIA_GPU_Computing_SDK/C/common/inc
CPPFLAGS 	+= `pkg-config --cflags opencv`
LIB 		+= `pkg-config --libs opencv` -L/home/wuhao/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 
OBJDIR		+= OBJ/
OBJS		+= $(addprefix $(OBJDIR), cudaSSD.o SAD.o SSD.o evaluate.o main.o)
#OBJS		+= OBJ/cudaSSD.o OBJ/SAD.o OBJ/SSD.o OBJ/evaluate.o OBJ/main.o

stereo: 	$(OBJS)
	nvcc $(OBJS) -o $@ $(LIB)

OBJ/main.o: 	main.cpp cudaSSD.h SAD.h SSD.h evaluate.h
	g++ -c  $< -o $@ $(CPPFLAGS)

OBJ/%.o: %.cpp
	g++  -c  $< -o $@ $(NVCCFLAGS)
OBJ/%.o : %.cu
	nvcc -c  $< -o $@ $(NVCCFLAGS)

clean:
	-rm stereo $(OBJS)
.PHONY: clean
