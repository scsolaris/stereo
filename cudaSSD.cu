#include "cudaSSD.h"

#include <cuda.h>
#include <cutil.h>
#include <stdio.h>
   
#define SQ(a) (__mul24(a,a));

int divUp(int a, int b)
{
	if(a%b == 0)
		return a/b;
	else
		return a/b + 1;
}

static int g_w;
static int g_h;

static float *g_disparityLeft;
static int *g_minSSD;
static size_t g_floatDispPitch;

cudaArray * g_leftTex_array;
cudaArray * g_rightTex_array;

unsigned int LeftImage_GLBufferID;
unsigned int RightImage_GLBufferID;
unsigned int DisparityImage_GLBufferID;

texture<unsigned char, 2, cudaReadModeNormalizedFloat> leftTex;
texture<unsigned char, 2, cudaReadModeNormalizedFloat> rightTex;

#define ROWSperTHREAD 40 // the number of rows a thread will process
#define BLOCK_W 64 // the thread block width
#define RADIUS_H 5 // Kernel Radius 5V & 5H = 11x11 kernel
#define RADIUS_V 5
#define MIN_SSD 500000 // The mimium acceptable SSD value
#define STEREO_MIND 0.0f // The minimum d range to check
#define STEREO_DISP_STEP 1.0f // the d step, must be <= 1 to avoid aliasing
#define SHARED_MEM_SIZE ((BLOCK_W + 2*RADIUS_H)*sizeof(int) ) // amount of 

__global__ void stereoKernel( float *disparityPixel,int *disparityMinSSD,int width,int height,size_t out_pitch,float STEREO_MAXD); 

void SetupStereo(unsigned int w, unsigned int h)
{
	g_w = w;
	g_h = h;

	cudaMallocPitch((void**)&g_disparityLeft,&g_floatDispPitch,w*sizeof(float),h);
	cudaMallocPitch((void**)&g_minSSD,&g_floatDispPitch,w*sizeof(int),h);
	g_floatDispPitch /= sizeof(float);

	cudaChannelFormatDesc U8Tex = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray(&g_leftTex_array, &U8Tex, g_w, g_h);
	cudaMallocArray(&g_rightTex_array, &U8Tex, g_w, g_h);
}

float cudaSSD(unsigned char * p_hostLeft, unsigned char * p_hostRight,unsigned char *Dispmap,int w,int h,float MAXD,int scale)
{
	SetupStereo(w,h);

	unsigned int timer;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));

	dim3 grid(1,1,1);
	dim3 threads(16,8,1);

	// Greyscale Image, just copy it.
	cudaMemcpyToArray(g_leftTex_array, 0, 0, p_hostLeft,g_w * g_h,
	cudaMemcpyHostToDevice);
	cudaMemcpyToArray(g_rightTex_array, 0, 0, p_hostRight,g_w * g_h,
	cudaMemcpyHostToDevice);

	// Set up the texture parameters for bilinear interpolation & clamping
	leftTex.filterMode = cudaFilterModeLinear;
	cudaBindTextureToArray(leftTex, g_leftTex_array);
	rightTex.filterMode = cudaFilterModeLinear;
	cudaBindTextureToArray(rightTex, g_rightTex_array);

	threads.x = BLOCK_W;
	threads.y = 1;
	grid.x = divUp(g_w, BLOCK_W);
	grid.y = divUp(g_h,ROWSperTHREAD);
	stereoKernel<<<grid,threads,SHARED_MEM_SIZE>>>(g_disparityLeft,g_minSSD,
	g_w,g_h,g_floatDispPitch,MAXD);

	cudaThreadSynchronize();
	cudaUnbindTexture(leftTex);
	cudaUnbindTexture(rightTex);
	CUT_SAFE_CALL(cutStopTimer(timer)); // don't time the drawing
	float retval = cutGetTimerValue(timer);

//	float* tDispmap = new float[g_floatDispPitch*g_h];
	float* tDispmap = new float[g_w*g_h];
	if(cudaSuccess!=cudaMemcpy2D(tDispmap,sizeof(float)*g_w,g_disparityLeft,sizeof(float)*g_floatDispPitch,g_w*sizeof(float),g_h,cudaMemcpyDeviceToHost)) printf("wrong!\n");
	for(int i=0;i<g_h;i++)
		for(int j=0;j<g_w;j++)
		{
//			printf("%f",tDispmap[i*g_floatDispPitch+j]);
			Dispmap[i*g_w+j] = (unsigned char)(tDispmap[i*g_w+j]*scale);
		}

	delete []tDispmap;
	return retval;	
}

__global__ void stereoKernel( float *disparityPixel,int *disparityMinSSD,int width,int height,size_t out_pitch,float STEREO_MAXD)
{
	extern __shared__ int col_ssd[]; // column squared difference functions
	float d; // disparity value
	int diff; // difference temporary value
	int ssd; // total SSD for a kernel
	float x_tex; // texture coordinates for image lookup
	float y_tex;
	int row; // the current row in the rolling window
	int i; // for index variable

	// use define¡¯s to save registers
	#define X (__mul24(blockIdx.x,BLOCK_W) + threadIdx.x)
	#define Y (__mul24(blockIdx.y,ROWSperTHREAD))

	// for threads reading the extra border pixels, this is the offset
	// into shared memory to store the values
	int extra_read_val = 0;
	if(threadIdx.x < (2*RADIUS_H)) extra_read_val = BLOCK_W+threadIdx.x;
	// initialize the memory used for the disparity and the disparity difference
	if(X<width )
	{
		for(i = 0;i < ROWSperTHREAD && Y+i < height;i++)
		{
			disparityPixel[__mul24((Y+i),out_pitch)+X] = -1;
//			disparityDiff[__mul24((Y+i),out_pitch)+X] = MIN_DISP;
			disparityMinSSD[__mul24((Y+i),out_pitch)+X] = MIN_SSD;
		}
	}
	__syncthreads();

	if( X < (width+RADIUS_H) && Y <= (height) )
	{
		x_tex = X - RADIUS_H;
		for(d = STEREO_MIND; d <= STEREO_MAXD; d += STEREO_DISP_STEP)
		{
			col_ssd[threadIdx.x] = 0;
			if(extra_read_val>0) col_ssd[extra_read_val] = 0;
			// do the first rows
			y_tex = Y - RADIUS_V;
			for(i = 0; i <= 2*RADIUS_V; i++)
			{
				diff = (int)(255.0f*tex2D(leftTex,x_tex,y_tex)) - (int)(255.0f*tex2D(rightTex,x_tex-d,y_tex));
				col_ssd[threadIdx.x] += SQ(diff);
				if(extra_read_val > 0)
				{
					diff = (int)(255.0f*tex2D(leftTex,x_tex+BLOCK_W,y_tex)) - (int)(255.0f*tex2D(rightTex,x_tex+BLOCK_W-d,y_tex));
					col_ssd[extra_read_val] += SQ(diff);
				}
				y_tex += 1.0f;
			}
			__syncthreads();
			// now accumulate the total
			if(X < width && Y < height)
			{
				ssd = 0;
				for(i = 0;i<(2*RADIUS_H);i++)
				{
					ssd += col_ssd[i+threadIdx.x];
				}
				if(ssd < disparityMinSSD[__mul24(Y,out_pitch) + X])
				{
					disparityPixel[__mul24(Y,out_pitch) + X] = d;
					disparityMinSSD[Y*out_pitch + X] = ssd;
				}
			}
			__syncthreads();
			// now do the remaining rows
			y_tex = (float)(Y - RADIUS_V); // this is the row we will remove
			for(row = 1;row < ROWSperTHREAD && (row+Y < (height+RADIUS_V)); row++)
			{
				// subtract the value of the first row from column sums
				diff = (int)(255.0f*tex2D(leftTex,x_tex,y_tex)) - (int)(255.0f*tex2D(rightTex,x_tex-d,y_tex));
				col_ssd[threadIdx.x] -= SQ(diff);
				// add in the value from the next row down
				diff = (int)(255.0f*tex2D(leftTex,x_tex,y_tex + (float)(2*RADIUS_V)+1.0f)) -
					(int)(255.0f*tex2D(rightTex,x_tex-d,y_tex +(float)(2*RADIUS_V)+1.0f));
				col_ssd[threadIdx.x] += SQ(diff);
				if(extra_read_val > 0)
				{
					diff = (int)(255.0f*tex2D(leftTex,x_tex+(float)BLOCK_W,y_tex)) -
					(int)(255.0f*tex2D(rightTex,x_tex-d+(float)BLOCK_W,y_tex));
					col_ssd[threadIdx.x+BLOCK_W] -= SQ(diff);
					diff = (int)(255.0f*tex2D(leftTex,x_tex+(float)BLOCK_W,y_tex +
					(float)(2*RADIUS_V)+1.0f)) -
					(int)(255.0f*tex2D(rightTex,x_tex-d+(float)BLOCK_W,y_tex +
					(float)(2*RADIUS_V)+1.0f));
					col_ssd[extra_read_val] += SQ(diff);
				}
				y_tex += 1.0f;
				__syncthreads();
				if(X<width && (Y+row) < height)
				{
					ssd = 0;
					for(i = 0;i<(2*RADIUS_H);i++)
					{
						ssd += col_ssd[i+threadIdx.x];
					}
					if(ssd < disparityMinSSD[__mul24(Y+row,out_pitch) + X])
					{
						disparityPixel[__mul24(Y+row,out_pitch) + X] = d;
						disparityMinSSD[__mul24(Y+row,out_pitch) + X] = ssd;
					}
				}
				__syncthreads(); // wait for everything to complete
			} // for row loop
		} // for d loop
	} // if 'int the image' loop
}
