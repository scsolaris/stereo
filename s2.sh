nvcc  cudaSSD.cu SAD.cpp -c -I/home/wuhao/NVIDIA_GPU_Computing_SDK/C/common/inc
g++ `pkg-config --cflags opencv` main.cpp -c
nvcc `pkg-config --libs opencv` cudaSSD.o SAD.o main.o -o stereo_32 -L/usr/local/cuda/lib -lcudart -L/usr/lib/nvidia-current -lcuda -L/home/wuhao/NVIDIA_GPU_Computing_SDK/C/common/lib/linux  -L/home/wuhao/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_i386 

