#include "Model.hpp"

class ModelDeeplab: public Model {
private:
    dnn::Net net;
public:

    ModelDeeplab(const string& model_path) {
        net = dnn::readNetFromTensorflow(model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
    Mat_<Vec3b> get_overlay(const Mat_<Vec3b>& input) override {    
        // Preprocess the image
        Mat blob;
        dnn::blobFromImage(input, blob, 0.007843, Size(513, 513), Scalar(127.5, 127.5, 127.5), true, false); // Resize and normalize

        // Set the input to the network
        net.setInput(blob);

        // Run the forward pass
        Mat output = net.forward();

        // Postprocess the output
        int rows = output.size[2];
        int cols = output.size[3];
        Mat_<uint8_t> segmentation_map(rows, cols);
        int nclasses=output.size[1];

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
                segmentation_map(i, j) = (uchar)(max_index * 255 / (output.size[1] - 1)); // Map class index to grayscale value
            }
        }
        // Resize the segmentation map to the original image size
        resize(segmentation_map, segmentation_map, input.size(), 0, 0, INTER_LINEAR);

        // Apply a color map for better visualization (optional)
        Mat colored_segmentation_map;
        applyColorMap(segmentation_map, colored_segmentation_map, COLORMAP_JET);
        addWeighted(input, 0.5, colored_segmentation_map, 0.5, 0.0, colored_segmentation_map);
        return colored_segmentation_map;
    }
};