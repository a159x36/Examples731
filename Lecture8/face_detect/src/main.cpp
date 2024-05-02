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


int main(int argc, char** argv) {
    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;
    Mat gray;
    CascadeClassifier cascade, nestedCascade;
    bool YUNet=false;

    Ptr<FaceDetectorYN> ynmodel;

    // Choose the correct one of these for your version of opencv
    //ynmodel=FaceDetectorYN::create("../../data/face_detection_yunet_2021sep.onnx","",Size(640,480)); // for v4.5
    ynmodel=FaceDetectorYN::create("../../data/face_detection_yunet_2022mar.onnx","",Size(640,480)); // for v4.6
    //ynmodel=FaceDetectorYN::create("../../data/face_detection_yunet_2023mar.onnx","",Size(640,480)); // for v4.8


    Menu m("Object Detection",{"Face","Face1","Face2","Body",
            "Upper Body","Cat","Eye","Eye1","Left Eye","Right Eye","Smile","YUNet"});

    string PATH="../../data/haarcascades/";
    string FACES=PATH+"haarcascade_frontalface_default.xml";
    string FACES1=PATH+"haarcascade_frontalface_alt.xml";
    string FACES2=PATH+"haarcascade_frontalface_alt_tree.xml";
    string EYES=PATH+"haarcascade_eye_tree_eyeglasses.xml";
    string EYES1=PATH+"haarcascade_eye.xml";
    string LEFTEYE=PATH+"haarcascade_lefteye_2splits.xml";
    string RIGHTEYE=PATH+"haarcascade_righteye_2splits.xml";
    string UPPERBODY=PATH+"haarcascade_upperbody.xml";
    string FULLBODY=PATH+"haarcascade_fullbody.xml";
    string CAT=PATH+"haarcascade_frontalcatface_extended.xml";
    string SMILE=PATH+"haarcascade_smile.xml";

    cascade.load(FACES1);
    nestedCascade.load(EYES);

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
    
    namedWindow("WebCam", WINDOW_FREERATIO);
    resizeWindow("WebCam",frame.cols,frame.rows);
    ynmodel->setInputSize(frame.size());

    float fps;
    while (1) {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if(!YUNet) {
            vector<Rect> faces;
            cascade.detectMultiScale( gray, faces, 1.1, 3, 0, Size(30, 30) );
            Mat roi;
            for ( Rect r:faces ) {
                rectangle( frame, r, Scalar(0,255,0), 2, LINE_AA);
                roi = gray( r );
                vector<Rect> nestedObjects;
                nestedCascade.detectMultiScale( roi, nestedObjects, 1.1, 3, 0, Size(10, 10) );
                for(Rect r1:nestedObjects) {
                    r1=r1+Point(r.x,r.y);
                    rectangle( frame, r1, Scalar(255,0,0), 2, LINE_AA);
                }
            }
        } else {
            Mat_<float> faces;
            static Scalar box_color{0, 255, 0};
            static vector<Scalar> landmark_color{
                Scalar(255,   0,   0), // right eye
                Scalar(  0,   0, 255), // left eye
                Scalar(  0, 255,   0), // nose tip
                Scalar(255,   0, 255), // right mouth corner
                Scalar(  0, 255, 255)  // left mouth corner
            };
            static Scalar text_color{0, 255, 0};
            ynmodel->detect(frame,faces);
            for (int i = 0; i < faces.rows; ++i) {
                // Draw bounding boxes
                int x1 = faces(i, 0);
                int y1 = faces(i, 1);
                int w = faces(i, 2);
                int h = faces(i, 3);
                rectangle(frame, Rect(x1, y1, w, h), box_color, 2);

                // Confidence as text
                float conf = faces(i, 14);
                putText(frame, format("%.4f", conf), Point(x1, y1+12), FONT_HERSHEY_DUPLEX, 0.5, text_color);

                // Draw landmarks
                for (int j = 0; j < landmark_color.size(); ++j) {
                    int x = faces(i, 2*j+4), y = faces(i, 2*j+5);
                    circle(frame, Point(x, y), 2, landmark_color[j], 2);
                }
            }
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

        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        fps = 1000.0f / mseconds;
        string s=m.getselected();
        if(s!="") YUNet=false;
        if(s=="Eye") nestedCascade.load(EYES);
        if(s=="Eye1") nestedCascade.load(EYES1);
        if(s=="Left Eye") nestedCascade.load(LEFTEYE);
        if(s=="Right Eye") nestedCascade.load(RIGHTEYE);
        if(s=="Face") cascade.load(FACES);
        if(s=="Face1") cascade.load(FACES1);
        if(s=="Face2") cascade.load(FACES2);
        if(s=="Body") cascade.load(FULLBODY);
        if(s=="Cat") cascade.load(CAT);
        if(s=="Upper Body") cascade.load(UPPERBODY);
        if(s=="Smile") nestedCascade.load(SMILE);
        if(s=="YUNet") YUNet=true;
        m.setselected(-1);
    }
}
