#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/xobjdetect.hpp> 
#include "menu.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

struct KnownFace {
    string name;
    Mat features;
};

static vector<Scalar> landmark_color{
    Scalar(255,   0,   0), // right eye
    Scalar(  0,   0, 255), // left eye
    Scalar(  0, 255,   0), // nose tip
    Scalar(255,   0, 255), // right mouth corner
    Scalar(  0, 255, 255)  // left mouth corner
};

int main(int argc, char** argv) {
    (void)argv[argc - 1];
    VideoCapture cap;
    Mat_<Vec3b> frame;

    bool showLandmarks = false;
    bool recognizeFaces = true;

    Ptr<FaceDetectorYN> ynmodel;
    Ptr<FaceRecognizerSF> recognizer;

    ynmodel = FaceDetectorYN::create("../../data/face_detection_yunet_2023mar.onnx","",Size(640,480)); // for v4.8 and 5.0
    recognizer = FaceRecognizerSF::create("../../data/face_recognition_sface_2021dec.onnx","", dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU); // for v4.8 and 5.0

    Menu m("Faces",{"Face Detection","Face Recognition","Show Landmarks"});

    vector<KnownFace> knownFaces;
    vector<string> knownFaceNames = {"Martin","Daniel","Mahdi"};

    for (string name: knownFaceNames) {
        KnownFace kf;
        kf.name = name;
        Mat testface = imread("../../../../Images/" + name + ".png");
        ynmodel->setInputSize(testface.size());
        Mat result;
        ynmodel->detect(testface,result);
        Mat cropped;
        recognizer->alignCrop(testface,result.row(0),cropped);
        recognizer->feature(cropped,kf.features);
        knownFaces.push_back(kf);
    }

    cap.open(0);
    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 1600);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    //   cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    //   cap.set(CAP_PROP_FRAME_WIDTH, 960);
    cap >> frame;
    
    namedWindow("WebCam", WINDOW_FREERATIO);
    resizeWindow("WebCam",frame.cols,frame.rows);
    ynmodel->setInputSize(frame.size());
    ynmodel->setNMSThreshold(0.1);
    ynmodel->setScoreThreshold(0.2);

    while (1) {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if (frame.empty()) break;

        Mat_<float> faces;
        ynmodel->detect(frame,faces);
        for(int i = 0; i < faces.rows; i++) {
            // Draw landmarks
            if(showLandmarks) {
                for (size_t j = 0; j < landmark_color.size(); ++j) {
                    int x = faces(i, 2*j+4), y = faces(i, 2*j+5);
                    circle(frame, Point(x, y), 2, landmark_color[j], 2);
                }
            }

            Mat aligned;
            recognizer->alignCrop(frame, faces.row(i), aligned);
            Mat feature;
            recognizer->feature(aligned, feature);
            bool recognized = false;
            if(recognizeFaces) {
            for(KnownFace face: knownFaces) {
                double similarity = recognizer->match(face.features, feature);
                if(similarity > 0.363) {
                    recognized = true;
                    rectangle(frame, Rect(faces(i, 0), faces(i, 1), faces(i, 2), faces(i, 3)), Scalar(0, 255, 0), 2);
                    putText(frame, format("%s: %.2f", face.name.c_str(), similarity), Point(faces(i, 0), faces(i, 1) - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
                } 
            }
            }
            if(!recognized) {
                rectangle(frame, Rect(faces(i, 0), faces(i, 1), faces(i, 2), faces(i, 3)), Scalar(0, 0, 255), 2);
                if(recognizeFaces)
                    putText(frame, "Unknown", Point(faces(i, 0), faces(i, 1) - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 0, 255), 2);
            }
        }


        system_clock::time_point end = system_clock::now();
        float mseconds = (end - start) / 1ms;
        float fps = 1000.0f / mseconds;
        ostringstream ss;
        ss<<setw(4)<<setprecision(4)<<setfill('0')<<fps<<"fps ";
        string str=ss.str();
        putText(frame, str, Vec2i(12, 32), FONT_HERSHEY_SIMPLEX, 1, Vec3i(0, 0, 0), 2);
        putText(frame, str, Vec2i(10, 30), FONT_HERSHEY_SIMPLEX, 1, Vec3i(0, 0, 255), 2);
        string s=m.getselected();
        if(s=="Face Detection") {
            showLandmarks = false;
            recognizeFaces = false;
        } else if(s=="Face Recognition") {
            showLandmarks = false;
            recognizeFaces = true;
        } else if(s=="Show Landmarks") {
            showLandmarks = true;
        }
        imshow("WebCam", frame);
        char c = (char)waitKey(1);
        if( c == 27 || c=='q')
            break;
    }
}
