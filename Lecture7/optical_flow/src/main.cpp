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

void showflow(Mat flow) {
    Mat flow_uv[2];
    Mat mag, ang, rgb;
    Mat hsv_split[3], hsv;
    split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    imshow("flow", rgb);
}

int main(int argc, char** argv) {
    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    Mat gray, prevGray;
    
        cap.open(0);
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
    
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    Menu m("Optical Flow",{"Lukas-Kanade","FarneBack","DIS","Exit"});
    m.setselected(0);
    const int MAX_COUNT = 500;
    
    namedWindow("WebCam", WINDOW_FREERATIO);
    resizeWindow("WebCam",frame.cols,frame.rows);

    vector<Point2f> points[2];
    Ptr<DenseOpticalFlow> dis = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    Mat_<Vec2f> flow(frame.size());
    float fps;
    while (1) {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if(prevGray.empty())
            gray.copyTo(prevGray);

        string method=m.getselected();
        if(method=="Lukas-Kanade") {
            goodFeaturesToTrack(gray, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 0, false, 0.04);
            cornerSubPix(gray, points[0], subPixWinSize, Size(-1,-1), termcrit);
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i;
            for( i =0; i < points[1].size(); i++ ) {
                circle(frame, points[1][i], 2, Scalar(255,0,0), 1, LINE_AA);
                line(frame, points[0][i], points[1][i],Scalar(0,255,0), 1, LINE_AA);
            }
        }
        if(method=="FarneBack") {
            calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            showflow(flow);
        }
        if(method=="DIS") {
            dis->calc(prevGray, gray, flow);
            showflow(flow);
        }
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps ";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_PLAIN, 2, Vec3i(0, 0, 0), 2, 8);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_PLAIN, 2, Vec3i(0, 0, 255), 2, 8);
        imshow("WebCam", frame);

        char c = (char)waitKey(1);
        if( c == 27 || c=='q')
            break;

        cv::swap(prevGray, gray);
        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
    }
}
