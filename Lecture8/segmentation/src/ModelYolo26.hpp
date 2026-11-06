#include "Model.hpp"

class ModelYolo26: public Model {
private:
    dnn::Net net;
public:
    ModelYolo26(const string& model_path) {
        net = dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    }
    Mat_<Vec3b> get_overlay(const Mat_<Vec3b>& input) override {   
        Mat blob = dnn::blobFromImage(input, 1.0 / 255.0, Size(640,640), Scalar(), true, false, CV_32F); 
        Mat_<Vec3b> image=input.clone();
        net.setInput(blob);
        vector<Mat> outputs;
        net.forward(outputs);
        Mat_<float> output_3d = outputs[0];
        Mat_<float> mask_4d = outputs[1];
        // outputs have an extra dimension of size 1 at the front, so we can ignore it and treat them as 2D and 3D matrices, respectively.
        Mat_<float> mask=Mat(3,&mask_4d.size[1],mask_4d.type(),(void *)mask_4d.data);
        Mat_<float> output=Mat(2,&output_3d.size[1],output_3d.type(),(void *)output_3d.data);

        float threshold=0.35;
        float xgain=float(input.cols)/640.0f;
        float ygain=float(input.rows)/640.0f;
        Mat_<Vec3b> overlay=input.clone();

        vector<int> classcounts(NCOCOCLASSES, 0);
        for (int i = 0; i < output.rows; ++i) {
            if (output(i,4) < threshold) continue;
            const int classId = int(output(i,5));
            if (classId < 0 || classId >= NCOCOCLASSES)
                throw runtime_error("Unexpected class ID in model output.");
            classcounts[classId]++;
            Mat_<float> output_map(mask.size[1], mask.size[2]);            

            for (int j = 0; j < output_map.rows; ++j) {
                for (int k = 0; k < output_map.cols; ++k) {
                    if(j<output(i,1)/4 || j>output(i,3)/4 || k<output(i,0)/4 || k>output(i,2)/4) {
                        output_map(j, k) = 0;
                        continue;
                    }
                    float s=0;
                    for (int c = 0; c < 32; ++c) {
                        s+= mask(c,j,k) * output(i,6+c);
                    }
                    output_map(j, k) = s;
                }
            }
        
            resize(output_map, output_map, Size(input.cols,input.rows), 0, 0, INTER_LINEAR);
            Mat_<uint8_t> output_map_8u(output_map.rows, output_map.cols);
            cv::threshold(output_map, output_map, 0.0, 1.0, THRESH_BINARY);
            convertScaleAbs(output_map, output_map_8u, 255.0);

            vector<vector<Point>> contours;
            const Scalar color((71 * classId + 60) % 256, (137 * classId + 120) % 256,
                                (199 * classId + 200) % 256);

            findContours(output_map_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
            drawContours(overlay, contours, -1, color, -1);
            drawContours(image, contours, -1, color, 2);

            const float x1 = clamp(float(output(i,0) * xgain), 0.0f, float(input.cols));
            const float y1 = clamp(float(output(i,1) * ygain), 0.0f, float(input.rows));
            const float x2 = clamp(float(output(i,2) * xgain), 0.0f, float(input.cols));
            const float y2 = clamp(float(output(i,3) * ygain), 0.0f, float(input.rows));
            if (x2 <= x1 || y2 <= y1) continue;
            const Point p1(int(nearbyint(x1)), int(nearbyint(y1)));
            const Point p2(int(nearbyint(x2)), int(nearbyint(y2)));
            const auto label = format("%s:%d %.2f", cocoClasses[classId].c_str(), classcounts[classId], output(i,4));
            rectangle(overlay, p1, p2, Scalar(40, 220, 40), 2);
            putText(overlay, label, Point(p1.x, max(p1.y - 8, 22)),
                FONT_HERSHEY_SIMPLEX, 0.65, Scalar(40, 220, 40), 2, LINE_AA);
        }
        addWeighted(image, 0.5, overlay, 0.5, 0.0, overlay);
        return overlay;
    }
};