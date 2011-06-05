stereo: 	cudaSSD.o SAD.o SSD.o evaluate.o main.o
	nvcc $^ -o $@ $(LIB)

main.o: 	main.cpp cudaSSD.h SAD.h SSD.h evaluate.h
	g++ -c  $< $(CPPFLAGS)
cudaSSD.o:	cudaSSD.cu cudaSSD.h
SAD.o:		SAD.cpp SAD.h
SSD.o:		SSD.cpp SSD.h
evaluate.o:	evaluate.cpp evaluate.h

%.o: %.cpp
	g++ -c  $< $(NVCCFLAGS)
%.o: %.cu
	nvcc -c  $< $(NVCCFLAGS)

NVCCFLAGS 	+= -I/home/wuhao/NVIDIA_GPU_Computing_SDK/C/common/inc
CPPFLAGS 	+= `pkg-config --cflags opencv`
LIB 		+= `pkg-config --libs opencv` -L/home/wuhao/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 

clean:
	-rm stereo *.o
.PHONY: clean
