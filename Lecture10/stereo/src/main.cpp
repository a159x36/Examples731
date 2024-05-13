#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "menu.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

// main, uses webcam to get a colour image, encoded it using the dct, quantizes this
// then dequantizes it and decodes it using an idct. 
int main(int argc, char** argv) {
    
    (void)argv[argc - 1];

    Mat image_l=imread("../../im0.png",IMREAD_GRAYSCALE);
    Mat image_r=imread("../../im1.png",IMREAD_GRAYSCALE);

    resize(image_l,image_l, cv::Size(), 0.25, 0.25);
    resize(image_r,image_r, cv::Size(), 0.25, 0.25);

    namedWindow("Disparity",WINDOW_AUTOSIZE);

    Mat disparity,disparity1;

    Ptr<StereoMatcher> stereo=StereoBM::create(96,7);
    Ptr<StereoMatcher> stereosg=StereoSGBM::create(0,96,7);

    stereo->compute(image_l,image_r,disparity);
    stereosg->compute(image_l,image_r,disparity1);
    imshow("Left",image_l);
    imshow("Right",image_r);
    imshow("Disparity",(disparity*32));
    imshow("DisparitySG",(disparity1)*32);

    
    waitKey(0);


}
