#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/stereo.hpp>
#include "menu.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

void showdisp(string name, Mat disparity) {
    Mat disp8,disp8_3c;
    double min,max;
    minMaxLoc(disparity,&min,&max);
 //   cout<<"maxmin"<<min<<","<<max<<endl;
    disparity.convertTo(disp8, CV_8U,255.0/max);
    applyColorMap(disp8, disp8_3c, COLORMAP_TURBO);
    imshow(name,disp8_3c);
}

int redosg=true;
int redost=true;

void setValSg(int pos, void *data) {
    int *v=(int *)data;
    *v=pos;
    redosg=true;
}

void setValSt(int pos, void *data) {
    int *v=(int *)data;
    *v=pos;
    redost=true;
}

void maketrackbar(string tbname, string winname, int maxval, TrackbarCallback cb, int *v) {
    createTrackbar( tbname, winname, NULL, maxval, cb, v);
    setTrackbarPos(tbname, winname, *v);
}
// main
int main(int argc, char** argv) {
    
    (void)argv[argc - 1];
    int p1=10;
    int p2=100;
    int bs=9;
    int stbs=9;
    int nd=132;
    int stnd=132;

    Mat image_l=imread("../../im0.png",IMREAD_GRAYSCALE);
    Mat image_r=imread("../../im1.png",IMREAD_GRAYSCALE);

    resize(image_l,image_l, cv::Size(), 0.25, 0.25);
    resize(image_r,image_r, cv::Size(), 0.25, 0.25);

    namedWindow("DisparitySGBM",WINDOW_AUTOSIZE);
    namedWindow("DisparityBM",WINDOW_AUTOSIZE);
    
    Mat disparity;
    Ptr<StereoMatcher> stereo=StereoBM::create(132,9);
    Ptr<StereoSGBM> stereosg=StereoSGBM::create(0,132,9,10,100);

    maketrackbar( "P1", "DisparitySGBM", 1000, setValSg, &p1);
    maketrackbar( "P2", "DisparitySGBM", 1000, setValSg, &p2);
    maketrackbar( "BS", "DisparitySGBM", 50, setValSg, &bs);
    maketrackbar( "nD", "DisparitySGBM", 500, setValSg, &nd);
    maketrackbar( "BS", "DisparityBM", 20, setValSt, &stbs);
    maketrackbar( "nD", "DisparityBM", 500, setValSt, &stnd);
    imshow("Left",image_l);
    imshow("Right",image_r);
    Ptr<stereo::QuasiDenseStereo> stereoqd = stereo::QuasiDenseStereo::create(image_l.size());
   // show default parameters
    cout<<stereoqd->Param.corrWinSizeX<<endl;
    cout<<stereoqd->Param.corrWinSizeY<<endl;
    cout<<stereoqd->Param.lkTemplateSize<<endl;
    cout<<stereoqd->Param.gftMaxNumFeatures<<endl;
    cout<<stereoqd->Param.gftQualityThres<<endl;

  // some parameters to try
  //  stereoqd->Param.gftMinSeperationDist=40;
  //  stereoqd->Param.lkTemplateSize=5;
  //  stereoqd->Param.gftMaxNumFeatures=1000;
    
  //  stereoqd->Param.corrWinSizeX=9;
  //  stereoqd->Param.corrWinSizeY=9;
  //  stereoqd->Param.neighborhoodSize=5;
    stereoqd->process(image_l,image_r);
    disparity = stereoqd->getDisparity();
    showdisp("DisparityQD",disparity);

    while(1) {
        if(redosg) {
            stereosg->setP1(p1);
            stereosg->setP2(p2);
            stereosg->setNumDisparities(nd+1);
            stereosg->setBlockSize(bs);
            stereosg->compute(image_l,image_r,disparity);
            showdisp("DisparitySGBM",disparity);
            redosg=false;
        }
        if(redost) {
            stereo->setBlockSize(stbs*2+5);
            stereo->setNumDisparities((stnd/16+1)*16);
            stereo->compute(image_l,image_r,disparity);
            showdisp("DisparityBM",disparity);
            redost=false;
        }
        int k=waitKey(50);
        if(k=='q') exit(0);
    }


}
