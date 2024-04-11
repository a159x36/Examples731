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
    bool COLOUR=true;
    bool WEBCAM=true;
    bool WEINER=true;

    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    Mat_<Vec3b> image;
    image=imread("../../../../Images/clown.tif");
    frame=image.clone();
    if(WEBCAM) {
        cap.open(0,CAP_V4L2);
        if (!cap.isOpened()) {
            cout << "Failed to open camera" << endl;
            return 0;
        }
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
        //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
        //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
        cap >> frame;
    }
    
    Menu m("Deconvolution",{"Weiner Filter","Pseudoinverse","Colour","Greyscale","Webcam","Static Image","Exit"});
    namedWindow("WebCam", WINDOW_FREERATIO);
    namedWindow("Result", WINDOW_FREERATIO);

    printf("frame size %d %d\n", frame.rows, frame.cols);
    resizeWindow("WebCam",frame.cols,frame.rows);
    resizeWindow("Result",frame.cols,frame.rows);

    int key = 0;
    float fps = 0.0;

    Mat_<Vec2f> complex1;
    Mat_<Vec2f> fft;
    Mat_<Vec3b> colour;
    Mat_<uint8_t> channels[3];
    Mat_<float> planes[2];

    int noise=108;
    int cutoff=138;//frame.cols/4;
    int focus=36;
    createTrackbar( "cutoff", "WebCam", &cutoff, (frame.rows+frame.cols)*2);
    createTrackbar( "noise", "WebCam", &noise, 1000);
    createTrackbar( "focus", "WebCam", &focus, 255);
    while (1) {
        system_clock::time_point start = system_clock::now();
        if(WEBCAM) {
            cap.set(CAP_PROP_FOCUS,focus);
            cap >> frame;
        } else {
            frame=image.clone();
        }
        if (frame.empty()) break;

        if(COLOUR) {
            split(frame,channels);
        } else {
            cvtColor(frame,colour,COLOR_BGR2YUV);    
            split(colour,channels);
        }
        
        
        int nchannels=1;
        if(COLOUR) nchannels=3;

        for(int ch=0;ch<nchannels;ch++) {
            planes[0]=Mat_<float>(channels[ch].clone());
            planes[1]=Mat::zeros(frame.size(),CV_32F);
            merge(planes, 2, complex1);
            dft(complex1,fft);

            float rad=(cutoff/100.0+1)/5;
            for(int i=0;i<frame.rows;i++) {
                for(int j=0;j<frame.cols;j++) {

                    int i1=i<frame.rows/2?i:frame.rows-i;
                    int j1=j<frame.cols/2?j:frame.cols-j;
                    float d=rad*hypot(i1,j1)/10;
                    float H,G;
                    if(d!=0)
                        H=sin(d)/(d);      
                    else
                        H=1;  
                    if(WEINER) {
                        //  G = H / ((|H|^2)+N)
                        float denom=H*H+noise/10000.0;
                        G=H/denom;
                    } else {
                        if(H>noise/1000.0)
                            G=1/H;
                        else
                            G=0;
                    }
                    fft(i,j)=fft(i,j)*G;
                }
            }
            
            idft(fft,complex1,DFT_SCALE);

            split(complex1,planes);
            planes[0].convertTo(channels[ch],CV_8UC1);
        }
    
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps ";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 255), 2, 8);
        imshow("WebCam", frame);
        if(!COLOUR) {
            imshow("Result",channels[0]);
        } else {
            merge(channels,3,colour);
            imshow("Result",colour);
        }
        
        
        key = waitKey(1);
        if (key == 113 || key == 27) return 0;  // either esc or 'q'
        string entry=m.getselected();
        if(entry=="Weiner Filter") {
            WEINER=true;
        }
        if(entry=="Pseudoinverse") {
            WEINER=false;
        }
        if(entry=="Colour") {
            COLOUR=true;
        }
        if(entry=="Greyscale") {
            COLOUR=false;
        }
        if(entry=="Webcam") {
            WEBCAM=true;
        }
        if(entry=="Static Image") {
            WEBCAM=false;
        }
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
    }
}
