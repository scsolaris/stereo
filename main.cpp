#include "SAD.h"
#include "SSD.h"
#include "cudaSSD.h"
#include "cudaSAD.h"
#include "evaluate.h"

#include "cv.h"
#include "highgui.h"
#include <stdio.h>

#define  IMAGEL      "backl.bmp"
#define  IMAGER      "backr.bmp"

int main()
{
    //initiation
	const char* imgname[4]={"Tsukuba","Venus","Teddy","Cones"};
	int MAX_Disp[4]={16,20,60,60};
	int scale[4]={16,8,4,4};	
	char imL[30],imR[30],imDisp[30],truedisp[30],non_occ[30];
	IplImage *pImgl,*pImgr,*pDisp,*pTruedisp,*pNon_occ;
	int w,h,step;	//width,height,widthStep
	unsigned char *left,*right,*disp,*groundtruth,*nonocc;
	float time,err_nonocc;

	for(int i=0;i<4;i++)
	{
		sprintf(imL,"img/%s/imL.png",imgname[i]);
		sprintf(imR,"img/%s/imR.png",imgname[i]);
		sprintf(truedisp,"img/%s/groundtruth.png",imgname[i]);
		sprintf(non_occ,"img/%s/nonocc.png",imgname[i]);
		printf("loading %s\n",imgname[i]);
		pImgl = cvLoadImage(imL,0);
		if(pImgl == NULL) printf("cannot open imgl!\n");
    	pImgr = cvLoadImage(imR,0);
	    if(pImgr == NULL) printf("cannot open imgr!\n");
    	pTruedisp = cvLoadImage(truedisp,0);
	    if(pTruedisp == NULL) printf("cannot open truedisp!\n");
    	pNon_occ = cvLoadImage(non_occ,0);
	    if(pNon_occ == NULL) printf("cannot open non_occ!\n");
		pDisp = cvCreateImage(cvGetSize(pImgl),pImgl->depth,pImgl->nChannels);

		w=pImgl->width;
		h=pImgl->height;
		step=pImgl->widthStep;
		left  = new unsigned char[w*h];
		right = new unsigned char[w*h];
		disp  = new unsigned char[w*h];
		groundtruth  = new unsigned char[w*h];
		nonocc  = new unsigned char[w*h];
	
		for(int y=0;y<h;y++)
			for(int x=0;x<w;x++)
			{
				left[y*w+x] =(unsigned char) pImgl->imageData[y*step+x];	
				right[y*w+x] =(unsigned char) pImgr->imageData[y*step+x];
				groundtruth[y*w+x] =(unsigned char) pTruedisp->imageData[y*step+x];
				nonocc[y*w+x] =(unsigned char) pNon_occ->imageData[y*step+x];
			}	

		//SAD
		time=SAD(left,right,disp,w,h,MAX_Disp[i],scale[i]);
		err_nonocc = eval_nonocc(groundtruth,disp,nonocc,w,h,scale[i]);	
		for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
				pDisp->imageData[y*step+x] = (char)disp[y*w+x];
		sprintf(imDisp,"img/%s/disp_SAD.png",imgname[i]);
	    cvSaveImage(imDisp,pDisp);
		printf("SAD algorithm\n");
		printf("proccessing time:\t%6.3fms\n",time);
		printf("Percentage of Bad Matching pixels:\t%6.3f%%\n",err_nonocc*100);
	
		//SSD
		time=SSD(left,right,disp,w,h,MAX_Disp[i],scale[i]);
		err_nonocc = eval_nonocc(groundtruth,disp,nonocc,w,h,scale[i]);	
		for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
				pDisp->imageData[y*step+x] = (char)disp[y*w+x];
		sprintf(imDisp,"img/%s/disp_SSD.png",imgname[i]);
	    cvSaveImage(imDisp,pDisp);
		printf("SSD algorithm\n");
		printf("proccessing time:\t%6.3fms\n",time);
		printf("Percentage of Bad Matching pixels:\t%6.3f%%\n",err_nonocc*100);
	
		//cudaSSD
		time = cudaSSD(left,right,disp,w,h,MAX_Disp[i],scale[i]);
		err_nonocc = eval_nonocc(groundtruth,disp,nonocc,w,h,scale[i]);	
		for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
				pDisp->imageData[y*step+x] = (char)disp[y*w+x];
		sprintf(imDisp,"img/%s/disp_cudaSSD.png",imgname[i]);
	    cvSaveImage(imDisp,pDisp);
		printf("cudaSSD algorithm\n");
		printf("proccessing time:\t%6.3fms\n",time);
		printf("Percentage of Bad Matching pixels:\t%6.3f%%\n",err_nonocc*100);

/*		//cudaSAD
		time = cudaSAD(left,right,disp,w,h,MAX_Disp[i],scale[i]);
		err_nonocc = eval_nonocc(groundtruth,disp,nonocc,w,h,scale[i]);	
		for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
				pDisp->imageData[y*step+x] = (char)disp[y*w+x];
		sprintf(imDisp,"img/%s/disp_cudaSAD.png",imgname[i]);
	    cvSaveImage(imDisp,pDisp);
		printf("cudaSAD algorithm\n");
		printf("proccessing time:\t%6.3fms\n",time);
		printf("Percentage of Bad Matching pixels:\t%6.3f%%\n",err_nonocc*100);
*/
		printf("\n");

		delete []left;
		delete []right;
		delete []disp;
	}

    return 0;
}
