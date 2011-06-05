#include "SAD.h"
#include <string.h>
#include <stdlib.h>
#include <cutil.h>

#define  W_R    5   //window radius

float SAD(unsigned char* left,unsigned char* right,unsigned char* Dispmap,int w,int h,int MAX_DISPARITY,int scale)
{
    //init
    int* col_sad = new int[w];
    int sad;
    int* min_sad = new int[w*h];
    unsigned char* tdisp = new unsigned char[w*h];
    memset(tdisp,0,sizeof(unsigned char)*w*h);
    int i,j;
    for(i=0;i<w*h;i++) min_sad[i]=500000;
	
	unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

    for(int d=0;d<MAX_DISPARITY;d++)
    {
        memset(col_sad,0,sizeof(int)*w);
        //do the first rows
        for(i=d;i<w;i++)    //calculate the col_sad
            for(j=0;j<(2*W_R+1);j++)    col_sad[i]+=abs((int)left[j*w+i]-(int)right[j*w+i-d]);
        sad=0;
        for(i=d;i<(2*W_R+d);i++) sad+=col_sad[i];   //sad of the first cols
        for(i=W_R+d;i<(w-W_R);i++)
        {
            if(i==W_R)
                sad+= col_sad[i+W_R];
            else
                sad += col_sad[i+W_R]-col_sad[i-W_R-1];
			if(sad < min_sad[W_R*w+i])
            {
                min_sad[W_R*w+i]=sad;
                tdisp[W_R*w+i]=d*scale;
            }
        }
        //do the remaining rows
        for(int row=W_R+1;row<h-W_R-1;row++)
        {
            for(i=d;i<w;i++) col_sad[i]+=abs((int)left[(row+W_R)*w+i]-(int)right[(row+W_R)*w+i-d])
                                        -abs((int)left[(row-W_R-1)*w+i]-(int)right[(row-W_R-1)*w+i-d]);
            sad=0;
            for(i=d;i<(2*W_R+d);i++) sad+=col_sad[i];   //sad of the first cols
            for(i=W_R+d;i<(w-W_R);i++)
            {
                if(i==W_R)
                    sad+= col_sad[i+W_R];
                else
                    sad += col_sad[i+W_R]-col_sad[i-W_R-1];
//              printf("s1%5.0f ",(double)sad);
                if(sad < min_sad[row*w+i])
                {
                    min_sad[row*w+i]=sad;
                    tdisp[row*w+i]=d*scale;
                }
            }
        }
    }
    memcpy(Dispmap,tdisp,sizeof(unsigned char)*w*h);
    delete []min_sad;
    delete []col_sad;
    delete []tdisp;
	CUT_SAFE_CALL(cutStopTimer(timer)); 
    float retval = cutGetTimerValue(timer);
	return retval ;
}
