nvcc  cudaSSD.cu SAD.cpp SSD.cpp -c -I/home/wuhao/NVIDIA_GPU_Computing_SDK/C/common/inc
g++ `pkg-config --cflags opencv` evaluate.cpp main.cpp -c
nvcc `pkg-config --libs opencv` cudaSSD.o SAD.o SSD.o evaluate.o main.o -o stereo -L/usr/local/cuda/lib64 -lcudart -L/usr/lib/nvidia-current -lcuda -L/home/wuhao/NVIDIA_GPU_Computing_SDK/C/common/lib/linux  -L/home/wuhao/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 
