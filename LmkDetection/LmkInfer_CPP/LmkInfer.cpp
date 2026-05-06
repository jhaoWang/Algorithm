#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

int main()
{
	const wchar_t* ONNX_PATH = L"E:\\pythonCode\\onnxInfer\\LmkDetection\\InferByPython\\landmark.onnx";
    const char* IMG_PATH = "E:\\pythonCode\\lafin-master\\lafin-master\\examples\\images\\195579.jpg";
    const char* MASK_PATH = "E:\\pythonCode\\lafin-master\\lafin-master\\examples\\masks\\08012.png";
    const char* SAVE_RESULT = "result_landmark.png";
    const int INPUT_SIZE = 256;
    const int LANDMARK_NUM = 68;

    // 初始化OnnxRuntime
    Env env(ORT_LOGGING_LEVEL_WARNING, "LandmarkInfer");
    SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Session ort_sess(env, ONNX_PATH, session_options);

    Mat src_img = imread(IMG_PATH);
    int h = src_img.rows;
    int w = src_img.cols;

    Mat img = imread(IMG_PATH);
    resize(img, img, Size(INPUT_SIZE, INPUT_SIZE));
    img.convertTo(img, CV_32FC3, 1.0 / 255.0f);

    // 转换输入格式
    vector<float> input_img;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                input_img.push_back(img.at<Vec3f>(y, x)[c]);
            }
        }
    }

    Mat mask = imread(MASK_PATH, IMREAD_GRAYSCALE);
    resize(mask, mask, Size(INPUT_SIZE, INPUT_SIZE));
    mask.convertTo(mask, CV_32FC1, 1.0 / 255.0f);

    vector<float> input_mask;
    for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            input_mask.push_back(mask.at<float>(y, x));
        }
    }

    // 原始图像添加掩码遮挡
    vector<float> input_feed(input_img.size());
    for (int i = 0; i < input_img.size(); i++) {
        int mask_idx = i % (INPUT_SIZE * INPUT_SIZE);
        float m = input_mask[mask_idx];
        input_feed[i] = input_img[i] * (1.0f - m) + m;
    }

    // 构造模型输入
    vector<const char*> input_names = { "onnx::Mul_0", "onnx::Sub_1" };
    vector<const char*> output_names = { "562" };
    vector<int64_t> input_feed_shape = { 1, 3, INPUT_SIZE, INPUT_SIZE };
    vector<int64_t> mask_shape = { 1, 1, INPUT_SIZE, INPUT_SIZE };

    vector<Value> input_tensors;
    input_tensors.push_back(Value::CreateTensor<float>(
        MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        input_feed.data(), input_feed.size(),
        input_feed_shape.data(), input_feed_shape.size()
    ));

    input_tensors.push_back(Value::CreateTensor<float>(
        MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        input_mask.data(), input_mask.size(),
        mask_shape.data(), mask_shape.size()
    ));

    // 模型推理
    vector<Value> outputs = ort_sess.Run(
        RunOptions(),
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size()
    );

    float* landmark_pred = outputs[0].GetTensorMutableData<float>();
    vector<Point2f> landmarks(LANDMARK_NUM);
    for (int i = 0; i < LANDMARK_NUM; i++) {
        float x = landmark_pred[i * 2];
        float y = landmark_pred[i * 2 + 1];

        // 等价Python：landmark_pred >= INPUT_SIZE-1 设为 INPUT_SIZE-1
        x = (x >= INPUT_SIZE - 1) ? (INPUT_SIZE - 1) : x;
        y = (y >= INPUT_SIZE - 1) ? (INPUT_SIZE - 1) : y;

        landmarks[i] = Point2f(x, y);
    }

    float scale_x = (float)w / INPUT_SIZE;
    float scale_y = (float)h / INPUT_SIZE;

    Mat black_background = Mat::zeros(h, w, CV_8UC1);
    for (auto& p : landmarks) {
        int ori_x = cvRound(p.x * scale_x);
        int ori_y = cvRound(p.y * scale_y);
        circle(black_background, Point(ori_x, ori_y), 2, Scalar(255), -1);
    }

    imshow("Landmark Result", black_background);
    imwrite(SAVE_RESULT, black_background);
    waitKey(0);
    destroyAllWindows();

    return 0;


}