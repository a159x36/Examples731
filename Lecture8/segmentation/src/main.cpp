#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Model.hpp"
#include "ModelDeepLab.hpp"
#include "ModelRFDetr.hpp"
#include "ModelYolo26.hpp"

#include "menu.hpp"

using namespace cv;
using namespace std;
using namespace chrono;


int main(int argc, char** argv) {

    string PATH="../../data/";
    string DEEPLABV3=PATH+"opt_deeplabv3_mnv2_513.pb";
    string YOLO26=PATH+"yolo26n-seg.onnx";
    string RFDETR=PATH+"rf_detr_seg.onnx";

    Model* model;

    Menu m("Segmentation",{"DeeplabV3","Yolo26","RFDetr","Exit"});
    
    m.setselected(0);
    string choice="";

    Mat image;
    VideoCapture cap;

    cap.open(0);
    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    while(1) {
        string modelname=m.getselected();
        if(choice!=modelname) {
            cout << "Switching to model: " << modelname << endl;
            choice=modelname;
            if(choice=="DeeplabV3") model=new ModelDeeplab(DEEPLABV3);
            if(choice=="Yolo26") model=new ModelYolo26(YOLO26);
            if(choice=="RFDetr") model=new ModelRFDetr(RFDETR);
        }
        cap >> image;
        Mat colored_segmentation_map=model->get_overlay(image);
        imshow("Original Image", image);
        imshow("Colored Segmentation Map", colored_segmentation_map);
        int k=waitKey(1);
        if(k==' ') break;
    }
}
