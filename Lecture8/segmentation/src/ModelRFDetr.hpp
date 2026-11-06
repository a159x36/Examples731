#include "Model.hpp"


class ModelRFDetr: public Model {
private:
    static constexpr int cocoIds[] = {
        1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
        21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,
        59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,
        80,81,82,84,85,86,87,88,89,90
    };
    dnn::Net net;
    map<int,string> cocoMap;
public:
    ModelRFDetr(const string& model_path) {
        net = dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        for(int id=0; id<NCOCOCLASSES; id++) cocoMap[cocoIds[id]]=cocoClasses[id];
    }
    Mat_<Vec3b> get_overlay(const Mat_<Vec3b>& input) override {
        Mat_<Vec3b> image=input.clone();
      //  float threshold = 0.5f;
        Mat_<float> blob=dnn::blobFromImage(input, 1.0/255.0, Size(312, 312), Scalar(0.485f, 0.456f, 0.406f), true, false);
        Mat_<float> blob_3d=Mat_<float>(3, &blob.size[1], blob.ptr<float>(0));
        float std_dev[] = {0.229f, 0.224f, 0.225f};
        
        for(int r=0;r<312;r++){
            for(int c=0;c<312;c++){
                for(int ch=0;ch<3;ch++){
                    blob_3d(ch ,r, c) /= std_dev[ch];
                }
            }
        }
        net.setInput(blob);
        std::vector<Mat_<float>> outputs;
        net.forward(outputs, std::vector<std::string>{"dets", "labels", "masks"});
        Mat_<float> labels=outputs[1];
        Mat_<float> masks=outputs[2];
        Mat_<float> dets=outputs[0];
        Size imageSize=input.size();
        Mat_<Vec3b> result = input.clone();
        for(int i=0;i<labels.size[1];i++){ // query
            for(int j=0;j<labels.size[2];j++){ // category
              //  
                if(labels(0,i,j)>0.0f) {//score>threshold) {
                    float score= 1.0f / (1.0f + std::exp(-std::clamp(labels(0,i,j), -80.0f, 80.0f)));
                    cout<< "query: " << i << " category: " << j << " score: " << score << "label: " << labels(0,i,j) << endl;
                    const float* b = dets.ptr<float>(0, i);
                    Vec4f box((b[0] - b[2] / 2) * imageSize.width,
                        (b[1] - b[3] / 2) * imageSize.height,
                        (b[0] + b[2] / 2) * imageSize.width,
                        (b[1] + b[3] / 2) * imageSize.height);
                    for (int i = 0; i < 4; ++i)
                        box[i] = std::clamp(box[i], 0.0f,
                                        float((i % 2 == 0 ? imageSize.width : imageSize.height) - 1));
                    Mat_<float> lowResolution(78, 78,
                                        const_cast<float*>(masks.ptr<float>(0, i)));
                    Mat_<float> resized;
                    Mat_<uchar> mask;
                    // Resize logits first, THEN threshold: logit > 0 means probability > 0.5.
                    resize(lowResolution, resized, imageSize, 0, 0, INTER_LINEAR);
                    compare(resized, 0.0, mask, CMP_GT);
                    const Scalar color((71 * j + 60) % 256, (137 * j + 120) % 256,
                                        (199 * j + 200) % 256);
                //    for (int y = 0; y < result.rows; ++y) {
                //        for (int x = 0; x < result.cols; ++x)
                //            if (mask(y,x))
                //                for (int c = 0; c < 3; ++c)
                //                    result(y,x)[c] = static_cast<unsigned char>(0.55 * result(y,x)[c] + 0.45 * color[c]);
                //    }
                    std::vector<std::vector<Point>> contours;
                    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                    drawContours(result, contours, -1, color, -1);
                    drawContours(image, contours, -1, color, 2);
                    const Point topLeft(cvRound(box[0]), cvRound(box[1]));
                    const Point bottomRight(cvRound(box[2]), cvRound(box[3]));
                    std::ostringstream label;
                    label << cocoMap[j] << ' ' << std::fixed
                        << std::setprecision(2) << score;
                    rectangle(result, topLeft, bottomRight, {255, 255, 255}, 1);
                    const Point textAt(topLeft.x, std::max(topLeft.y - 5, 18));
                   // putText(result, label.str(), textAt, FONT_HERSHEY_SIMPLEX,
                   //             0.6, {0, 0, 0}, 3, LINE_AA);
                    putText(result, label.str(), textAt, FONT_HERSHEY_SIMPLEX,
                                0.6, {255, 255, 255}, 1, LINE_AA);
                    std::cout << label.str() << '\n';

                }
            }
        }
        addWeighted(image, 0.5, result, 0.5, 0.0, result);
        return result;
    }
};