//#include "cv.h"
//#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>

#define  eval_bad_thresh 1.0	// Acceptable disparity error

float eval_nonocc(unsigned char* true_disp,unsigned char* disp,unsigned char* occ,int w,int h,int scale)
{
	float err_occ=0;
	int t,i,n=0;
	for(int y=0;y<h;y++)
        for(int x=0;x<w;x++)
        {
			i=y*w+x;
			if(occ[i])
			{
				n++;
				t=abs((int)((true_disp[i]-disp[i])/scale));
//				printf("%d,%d,%d ",true_disp[i],disp[i],t);
				if(t > eval_bad_thresh) err_occ++;	
			}		
		}

	err_occ/=n;
	return err_occ;	
}
