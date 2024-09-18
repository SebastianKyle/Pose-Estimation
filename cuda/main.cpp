#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

cv::VideoCapture cap(0);

struct Configs {
    cv::Size INPUT_SIZE = cv::Size(192, 256);
    float SIGMA = 20.0f;
    float DECODE_BETA = 150.0f;
    float VISIBILITY_CONFIDENCE = 0.38f;
};

inline const Configs g_Configs;

void decode_simcc_labels(
    const std::vector<cv::Mat>& x_labels, 
    const std::vector<cv::Mat>& y_labels, 
    int batch_size,
    int num_joints,
    int W,
    int H,
    float sigma,
    float decode_beta,
    std::vector<cv::Point2f>& joints,
    std::vector<float>& visibility
);

int main(int, char **)
{
    cv::cuda::printCudaDeviceInfo(0);

    // Read model from onnx format
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../../../../models/simcc_192x256.onnx"); // you may adjust the path to your needs
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::Mat img;
    cv::Mat input;
    cv::Mat blob;

    while (cap.isOpened())
    {
        auto start = cv::getTickCount();

        cap.read(img);
        input = img.clone();

        cv::resize(input, input, g_Configs.INPUT_SIZE);
        blob = cv::dnn::blobFromImage(input, 1.0/255.0, g_Configs.INPUT_SIZE, cv::Scalar(0, 0, 0), true, false);
        blob = blob.reshape(1, { 1, blob.size[1], blob.size[2], blob.size[3] });

        net.setInput(blob);
        std::vector< cv::Mat > outputBlobs;
        net.forward(outputBlobs, net.getUnconnectedOutLayersNames());

        cv::Mat simcc_x = outputBlobs[0];
        cv::Mat simcc_y = outputBlobs[1];

        int batch_size = simcc_x.size[0];
        int num_joints = simcc_x.size[1];
        int W = simcc_x.size[2];
        int H = simcc_y.size[2];

        std::vector<cv::Mat> x_labels, y_labels;
        for (int i = 0; i < batch_size; ++i) {
            cv::Mat x_batch = simcc_x.row(i).clone().reshape(1, 1);
            cv::Mat y_batch = simcc_y.row(i).clone().reshape(1, 1);
            x_labels.push_back(x_batch);
            y_labels.push_back(y_batch);
        }
        
        std::vector<cv::Point2f> joints;
        std::vector<float> visibility;

        decode_simcc_labels(
            x_labels,
            y_labels,
            batch_size,
            num_joints,
            W,
            H,
            g_Configs.SIGMA,
            g_Configs.DECODE_BETA,
            joints,
            visibility
        );

        for (size_t i = 0; i < joints.size(); ++i) {
            cv::Point2f joint = joints[i];
            joint.x *= img.size[1];
            joint.y *= img.size[0];

            if (visibility[i] > g_Configs.VISIBILITY_CONFIDENCE) {
                cv::circle(img, joint, 4, cv::Scalar(255, 100, 100), -1);  // Visible joint
            }
        }

        auto end = cv::getTickCount();

        auto totalTime = (end - start) / cv::getTickFrequency();
        auto fps = 1 / totalTime;

        std::cout << "\n FPS: " << fps << std::endl;
        cv::putText(img, "FPS: " + std::to_string(int(fps)), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 255), 1);

        cv::imshow("Image", img);

        // 'q' to quit
        int k = cv::waitKey(10);
        if (k == 113)
            break;
    }

    return 0;
}