#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            cout << msg << endl;
    }
} gLogger;

ICudaEngine* loadEngine(const string& enginePath) {
    ifstream file(enginePath, ios::binary);
    if (!file) { cout << "Cannot open engine file" << endl; return nullptr; }

    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), size);
    return engine;
}

int main() {
    string enginePath = "E:/code_git/Algorithm/FaceInpainting/trtInfer_python/inpaint.engine";
    ICudaEngine* engine = loadEngine(enginePath);
    IExecutionContext* context = engine->createExecutionContext();

    Mat img = imread("E:/pythonCode/lafin-master/lafin-master/examples/images/195579.jpg");
    Mat landmark = imread("E:/pythonCode/lafin-master/lafin-master/checkpoints/results/landmark_inpaint/landmark/195579.png", IMREAD_GRAYSCALE);
    Mat mask = imread("E:/pythonCode/lafin-master/lafin-master/examples/masks/11257.png", IMREAD_GRAYSCALE);

    resize(img, img, Size(256, 256));
    resize(landmark, landmark, Size(256, 256));
    resize(mask, mask, Size(256, 256));
    cvtColor(img, img, COLOR_BGR2RGB);

    threshold(mask, mask, 0, 255, THRESH_BINARY);

    Mat img_float, landmark_float, mask_float;
    img.convertTo(img_float, CV_32F, 1.0 / 127.5, -1.0);
    landmark.convertTo(landmark_float, CV_32F, 1.0 / 255.0);
    mask.convertTo(mask_float, CV_32F, 1.0 / 255.0);

    vector<float> input_img(1 * 3 * 256 * 256);
    vector<float> input_landmark(1 * 1 * 256 * 256);
    vector<float> input_mask(1 * 1 * 256 * 256);

    for (int h = 0; h < 256; h++) {
        for (int w = 0; w < 256; w++) {
            Vec3f pix = img_float.at<Vec3f>(h, w);
            input_img[0 * 256 * 256 + h * 256 + w] = pix[0];
            input_img[1 * 256 * 256 + h * 256 + w] = pix[1];
            input_img[2 * 256 * 256 + h * 256 + w] = pix[2];
        }
    }

    for (int h = 0; h < 256; h++)
        for (int w = 0; w < 256; w++)
            input_landmark[h * 256 + w] = landmark_float.at<float>(h, w);

    for (int h = 0; h < 256; h++)
        for (int w = 0; w < 256; w++)
            input_mask[h * 256 + w] = mask_float.at<float>(h, w);

    void* buffers[4];
    vector<float> output(1 * 3 * 256 * 256);

    cudaMalloc(&buffers[0], 3 * 256 * 256 * sizeof(float));
    cudaMalloc(&buffers[1], 1 * 256 * 256 * sizeof(float));
    cudaMalloc(&buffers[2], 1 * 256 * 256 * sizeof(float));
    cudaMalloc(&buffers[3], 3 * 256 * 256 * sizeof(float));

    cudaMemcpy(buffers[0], input_img.data(), input_img.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[1], input_landmark.data(), input_landmark.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[2], input_mask.data(), input_mask.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->executeV2(buffers);
    cudaMemcpy(output.data(), buffers[3], output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    Mat res(256, 256, CV_32FC3);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 256; h++) {
            for (int w = 0; w < 256; w++) {
                float val = output[c * 256 * 256 + h * 256 + w] * 255.0f;

                if (val < 0) val = 0;
                if (val > 255) val = 255;

                res.at<Vec3f>(h, w)[c] = val;
            }
        }
    }

    res.convertTo(res, CV_8UC3);
    cvtColor(res, res, COLOR_RGB2BGR);

    imshow("TRTResult", res);
    waitKey(0);
    imwrite("result_final.jpg", res);

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);

    cout << "Inference success！" << endl;
    return 0;
}