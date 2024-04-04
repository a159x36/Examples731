#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "menu.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

void satcallback(int click, void *object) {
    VideoCapture *cap=(VideoCapture *)object;
    bool supported=cap->set(CAP_PROP_SATURATION,click*1.0);
    if(!supported)
        cout<<"Property not supported"<<endl;
}
void brightnesscallback(int click, void *object) {
    VideoCapture *cap=(VideoCapture *)object;
    bool supported=cap->set(CAP_PROP_BRIGHTNESS,click*1.0);
    if(!supported)
        cout<<"Property not supported"<<endl;
}
void contrastcallback(int click, void *object) {
    VideoCapture *cap=(VideoCapture *)object;
    bool supported=cap->set(CAP_PROP_CONTRAST,click*1.0);
    if(!supported)
        cout<<"Property not supported"<<endl;
}
const int upB=23;
const int loB=0;
const int upG=168;
const int loG=37;
const int upR=135;
const int loR=40;

int marker_upB=upB;
int marker_loB=loB;
int marker_upG=upG;
int marker_loG=loG;
int marker_upR=upR;
int marker_loR=loR;

int colourmodel = 0;

void Find_markers(Mat_<Vec3b> image) {
    if(colourmodel==1)
        cvtColor(image,image, COLOR_BGR2HSV);
    if(colourmodel==2)
        cvtColor(image,image, COLOR_BGR2YUV);
    for(int r=0;r<image.rows;r++) {
        for(int c=0;c<image.cols;c++) {
            uint8_t blue=image(r,c)[0];
            uint8_t green=image(r,c)[1];
            uint8_t red=image(r,c)[2];
            if(blue<marker_loB || blue>marker_upB || 
            green<marker_loG || green>marker_upG || 
            red<marker_loR || red>marker_upR) {
                image(r,c)=0;
            } else 
                image(r,c)={255,255,255};
        }
    }
}
int main(int argc, char** argv) {
    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    cap.open(0,CAP_V4L2);
    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    
    Menu m("Colour Segmentation",{"RGB","HSV","YUV","Exit"});
    cout << "Opened camera:" << cap.get(CAP_PROP_BACKEND)<<endl;
    namedWindow("WebCam", WINDOW_FREERATIO);
    namedWindow("result", WINDOW_FREERATIO);
    
    createTrackbar("Saturation","WebCam",0,255,satcallback,&cap);
    createTrackbar("Brightness","WebCam",0,255,brightnesscallback,&cap);
    createTrackbar("Contrast","WebCam",0,255,contrastcallback,&cap);
    setTrackbarPos("Saturation","WebCam",100);
    setTrackbarPos("Brightness","WebCam",100);
    setTrackbarPos("Contrast","WebCam",100);
    createTrackbar( "up1", "result", &marker_upB, 255);
    setTrackbarPos("up1","result",upB);
    createTrackbar( "lo1", "result", &marker_loB, 255);
    setTrackbarPos("lo1","result",loB);

    createTrackbar( "up2", "result", &marker_upG, 255);
    setTrackbarPos("up2","result",upG);
    createTrackbar( "lo2", "result", &marker_loG, 255);
    setTrackbarPos("lo2","result",loG);

    createTrackbar( "up3", "result", &marker_upR, 255);
    setTrackbarPos("up3","result",upR);
    createTrackbar( "lo3", "result", &marker_loR, 255);
    setTrackbarPos("lo3","result",loR);

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
    cap >> frame;
    printf("frame size %d %d\n", frame.rows, frame.cols);
    resizeWindow("WebCam",frame.cols*1.5,frame.rows*1.5);
    resizeWindow("result",frame.cols*2,frame.rows*2);
    int key = 0;
    float fps = 0.0;
    Mat_<uint8_t> proc(frame.rows,frame.cols);

    while (1) {
        system_clock::time_point start = system_clock::now();
        
        cap >> frame;
        if (frame.empty()) break;
        Mat proc=frame.clone();
        Find_markers(proc);
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2,
                Vec3i(0, 0, 255), 2, 8);
        imshow("WebCam", frame);
        imshow("result",proc);
        key = waitKey(1);
        if (key == 113 || key == 27) return 0;  // either esc or 'q'
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
        string col=m.getselected();
        if(col=="RGB") colourmodel=0;
        if(col=="HSV") colourmodel=1;
        if(col=="YUV") colourmodel=2;
    }
}
