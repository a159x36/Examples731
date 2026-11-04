#include "Model.hpp"


class ModelRFDetr: public Model {
private:
    struct Instance {
        int query, category;
        float score;
        Vec4f box;  // x1, y1, x2, y2 in original-image coordinates.
        Mat_<uint8_t> mask;  // CV_8U, zero or 255.
    };
    static constexpr int cocoIds[] = {
        1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,
        21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,
        59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,
        80,81,82,84,85,86,87,88,89,90
    };
    dnn::Net net;
    map<int,string> cocoMap;

    std::vector<Instance> decode(const std::vector<Mat_<float>>& outputs,
                             Size imageSize, float threshold) {
        std::vector<float> scores(100 * 91);
        for (int q = 0; q < 100; ++q) {
            const float* logits = outputs[1].ptr<float>(0, q);
            for (int c = 0; c < 91; ++c)
                scores[q * 91 + c] = 1.0f / (1.0f + std::exp(-std::clamp(logits[c], -80.0f, 80.0f)));
        }
        std::vector<int> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 100, indices.end(),
                        [&](int a, int b) {
                            return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
                        });
        std::vector<Instance> instances;
        for (int index : indices) {
            const int query = index / 91, category = index % 91;
            if (scores[index] < threshold) continue;
            const float* b = outputs[0].ptr<float>(0, query);
            Vec4f box((b[0] - b[2] / 2) * imageSize.width,
                        (b[1] - b[3] / 2) * imageSize.height,
                        (b[0] + b[2] / 2) * imageSize.width,
                        (b[1] + b[3] / 2) * imageSize.height);
            for (int i = 0; i < 4; ++i)
                box[i] = std::clamp(box[i], 0.0f,
                                float((i % 2 == 0 ? imageSize.width : imageSize.height) - 1));
            Mat lowResolution(78, 78, CV_32F,
                                const_cast<float*>(outputs[2].ptr<float>(0, query)));
            Mat resized, mask;
            // Resize logits first, THEN threshold: logit > 0 means probability > 0.5.
            resize(lowResolution, resized, imageSize, 0, 0, INTER_LINEAR);
            compare(resized, 0.0, mask, CMP_GT);
            instances.push_back({query, category, scores[index], box, mask});
        }
        return instances;  // RF-DETR does not need NMS.
    }

    Mat_<Vec3b> drawInstances(const Mat_<Vec3b>& image, const std::vector<Instance>& instances) {
        Mat_<Vec3b> result = image.clone();
        cout << "Found " << instances.size() << " instances" << endl;
        for (size_t i = 0; i < instances.size(); ++i) {
            const Scalar color((71 * i + 60) % 256, (137 * i + 120) % 256,
                                (199 * i + 200) % 256);
            const Mat_<uint8_t>& mask = instances[i].mask;
            for (int y = 0; y < result.rows; ++y) {
                for (int x = 0; x < result.cols; ++x)
                    if (mask(y,x))
                        for (int c = 0; c < 3; ++c)
                            result(y,x)[c] = static_cast<unsigned char>(0.55 * result(y,x)[c] + 0.45 * color[c]);
            }
            std::vector<std::vector<Point>> contours;
            findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(result, contours, -1, color, 2);
        }
        for (const auto& instance : instances) {
            const auto& b = instance.box;
            const Point topLeft(cvRound(b[0]), cvRound(b[1]));
            const Point bottomRight(cvRound(b[2]), cvRound(b[3]));
            std::ostringstream label;
            label << cocoMap[instance.category] << ' ' << std::fixed
                << std::setprecision(2) << instance.score;
            rectangle(result, topLeft, bottomRight, {255, 255, 255}, 1);
            const Point textAt(topLeft.x, std::max(topLeft.y - 5, 18));
            putText(result, label.str(), textAt, FONT_HERSHEY_SIMPLEX,
                        0.6, {0, 0, 0}, 3, LINE_AA);
            putText(result, label.str(), textAt, FONT_HERSHEY_SIMPLEX,
                        0.6, {255, 255, 255}, 1, LINE_AA);
            std::cout << label.str() << '\n';
        }
        return result;
    }
public:
    ModelRFDetr(const string& model_path) {
        net = dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        for(int id=0; id<NCOCOCLASSES; id++) cocoMap[cocoIds[id]]=cocoClasses[id];
    }
    Mat_<Vec3b> get_overlay(const Mat_<Vec3b>& input) override {
        float threshold = 0.5f;
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
        const auto instances = decode(outputs, input.size(), threshold);
        const Mat result = drawInstances(input, instances);
        return result;
    }
};