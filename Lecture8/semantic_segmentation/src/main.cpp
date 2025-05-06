#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace chrono;
int main(int argc, char** argv) {

    string PATH="../../data/";
    string DEEPLABV3=PATH+"opt_deeplabv3_mnv2_513.pb";

    const char* class_names[]={"background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike","person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"};

    dnn::Net net = dnn::readNetFromTensorflow(DEEPLABV3);

    Mat image;
    VideoCapture cap;

    cap.open(0);
    if (!cap.isOpened()) {
        cout << "Failed to open camera" << endl;
        return 0;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    while(1) {
        cap >> image;
        // Preprocess the image
        Mat blob;
        dnn::blobFromImage(image, blob, 0.007843, Size(513, 513), Scalar(127.5, 127.5, 127.5), true, false); // Resize and normalize

        // Set the input to the network
        net.setInput(blob);

        // Run the forward pass
        Mat output = net.forward();

        // Postprocess the output
        int rows = output.size[2];
        int cols = output.size[3];
        Mat_<uint8_t> segmentation_map(rows, cols);
        int nclasses=output.size[1];
        // Find the class with the highest score for each pixel
        int classes[nclasses];
        for(int i=0;i<nclasses;i++)
            classes[i]=0;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float max_val = -1.0;
                int max_index = 0;
                for (int k = 0; k < nclasses; ++k) {
                    int p[4]={0,k,i,j};
                    float val = output.at<float>(p);
                    if (val > max_val) {
                        max_val = val;
                        max_index = k;
                    }
                }
                classes[max_index]++;
                segmentation_map(i, j) = (uchar)(max_index * 255 / (output.size[1] - 1)); // Map class index to grayscale value
            }
        }
        for(int i=0;i<nclasses;i++)
            if(classes[i]>0) cout<<class_names[i]<<":"<<classes[i]<<endl;
    
        // Resize the segmentation map to the original image size
        resize(segmentation_map, segmentation_map, image.size(), 0, 0, INTER_NEAREST);

        // Apply a color map for better visualization (optional)
        Mat colored_segmentation_map;
        applyColorMap(segmentation_map, colored_segmentation_map, COLORMAP_JET);

        // Blend the segmentation map with the original image (optional)
        Mat blended_image;
        addWeighted(image, 0.5, colored_segmentation_map, 0.5, 0.0, blended_image);

        // Display or save the results
        imshow("Original Image", image);
        imshow("Segmentation Map", segmentation_map);
        imshow("Colored Segmentation Map", colored_segmentation_map);
        imshow("Blended Image", blended_image);
        int k=waitKey(1);
        if(k==' ') break;
    }
}
