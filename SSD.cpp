#include "SSD.h"
#include <string.h>
#include <stdlib.h>
#include <cutil.h>

#define  W_R    5   //window radius

int square(int x)
{
	return x*x;
}
float SSD(unsigned char* left,unsigned char* right,unsigned char* Dispmap,int w,int h,int MAX_DISPARITY,int scale)
{
    //init
    int* col_ssd = new int[w];
    int ssd;
    int* min_ssd = new int[w*h];
    unsigned char* tdisp = new unsigned char[w*h];
    memset(tdisp,0,sizeof(unsigned char)*w*h);
    int i,j;
    for(i=0;i<w*h;i++) min_ssd[i]=500000;
	
	unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

    for(int d=0;d<MAX_DISPARITY;d++)
    {
        memset(col_ssd,0,sizeof(int)*w);
        //do the first rows
        for(i=d;i<w;i++)    //calculate the col_ssd
            for(j=0;j<(2*W_R+1);j++)    col_ssd[i]+=square((int)left[j*w+i]-(int)right[j*w+i-d]);
        ssd=0;
        for(i=d;i<(2*W_R+d);i++) ssd+=col_ssd[i];   //ssd of the first cols
        for(i=W_R+d;i<(w-W_R);i++)
        {
            if(i==W_R)
                ssd+= col_ssd[i+W_R];
            else
                ssd += col_ssd[i+W_R]-col_ssd[i-W_R-1];
			if(ssd < min_ssd[W_R*w+i])
            {
                min_ssd[W_R*w+i]=ssd;
                tdisp[W_R*w+i]=d*scale;
            }
        }
        //do the remaining rows
        for(int row=W_R+1;row<h-W_R-1;row++)
        {
            for(i=d;i<w;i++) col_ssd[i]+=square((int)left[(row+W_R)*w+i]-(int)right[(row+W_R)*w+i-d])
                                        -square((int)left[(row-W_R-1)*w+i]-(int)right[(row-W_R-1)*w+i-d]);
            ssd=0;
            for(i=d;i<(2*W_R+d);i++) ssd+=col_ssd[i];   //ssd of the first cols
            for(i=W_R+d;i<(w-W_R);i++)
            {
                if(i==W_R)
                    ssd+= col_ssd[i+W_R];
                else
                    ssd += col_ssd[i+W_R]-col_ssd[i-W_R-1];
//              printf("s1%5.0f ",(double)ssd);
                if(ssd < min_ssd[row*w+i])
                {
                    min_ssd[row*w+i]=ssd;
                    tdisp[row*w+i]=d*scale;
					if(tdisp[row*w+i]!=d*scale) printf("error!!!");
                }
            }
        }
    }
    memcpy(Dispmap,tdisp,sizeof(unsigned char)*w*h);
    delete []min_ssd;
    delete []col_ssd;
    delete []tdisp;
	CUT_SAFE_CALL(cutStopTimer(timer)); 
    float retval = cutGetTimerValue(timer);
	return retval ;
}
