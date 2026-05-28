#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>


using namespace std;
using namespace cv;
using namespace nvinfer1;

// ======================== 配置参数 ========================
const string ENGINE_PATH = "E:/code_git/Algorithm/YOLO/yolov5_detect/trtInfer_python/yolov5s.engine";
const string IMAGE_PATH = "E:/code_git/Algorithm/YOLO/yolov5_detect/trtInfer_python/bus.jpg";
const float CONF_THRESHOLD = 0.25f;
const float IOU_THRESHOLD = 0.45f;
const int INPUT_SIZE = 640;
const int NUM_CLASSES = 80;
const int OUTPUT_SIZE = 1 * 25200 * 85;
const int TEST_ITERATIONS = 50;

// ======================== TensorRT Logger ========================
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            cout << "[TRT] " << msg << endl;
    }
} gLogger;

// ======================== CUDA 错误检查 ========================
#define CHECK_CUDA(status) \
    if (status != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(status) << endl; \
        exit(-1); \
    }

// ======================== 加载 TRT 引擎 ========================
ICudaEngine* loadEngine(const string& enginePath) {
    ifstream file(enginePath, ios::binary);
    if (!file.good()) {
        cerr << "Engine file not found!" << endl;
        return nullptr;
    }

    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), size);
    return engine;  // TRT10 不用 destroy runtime
}

// ======================== 预处理 ========================
Mat preprocess(const Mat& img, int inputSize, float& scale, int& top, int& left) {
    int h = img.rows;
    int w = img.cols;
    scale = (float)inputSize / max(h, w);
    int newW = cvRound(w * scale);
    int newH = cvRound(h * scale);

    Mat resized;
    resize(img, resized, Size(newW, newH));

    Mat canvas = Mat::zeros(inputSize, inputSize, CV_8UC3);
    top = (inputSize - newH) / 2;
    left = (inputSize - newW) / 2;
    resized.copyTo(canvas(Rect(left, top, newW, newH)));

    canvas.convertTo(canvas, CV_32FC3, 1.0 / 255.0);
    cvtColor(canvas, canvas, COLOR_BGR2RGB);
    return canvas;
}

// ======================== 检测结果结构 ========================
struct Detection {
    float x1, y1, x2, y2;
    float score;
    int classId;
};

// ======================== 后处理 + NMS ========================
vector<Detection> postprocess(float* output, int oriH, int oriW, int inputSize,
    float confThresh, float iouThresh, float scale, int top, int left) {
    vector<Detection> detections;
    const int numBoxes = 25200;

    for (int i = 0; i < numBoxes; i++) {
        float* ptr = output + i * 85;
        float objConf = ptr[4];
        if (objConf < confThresh) continue;

        float classScore = 0;
        int classId = 0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (ptr[5 + j] > classScore) {
                classScore = ptr[5 + j];
                classId = j;
            }
        }

        float finalScore = objConf * classScore;
        if (finalScore < confThresh) continue;

        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        x1 = (x1 - left) / scale;
        y1 = (y1 - top) / scale;
        x2 = (x2 - left) / scale;
        y2 = (y2 - top) / scale;

        detections.push_back({ x1, y1, x2, y2, finalScore, classId });
    }

    vector<Rect> boxes;
    vector<float> scores;
    for (auto& det : detections) {
        boxes.emplace_back(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        scores.push_back(det.score);
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, scores, confThresh, iouThresh, indices);

    vector<Detection> res;
    for (int idx : indices) res.push_back(detections[idx]);
    return res;
}

// ======================== 主函数 ========================
int main() {
    // 1. 加载引擎
    ICudaEngine* engine = loadEngine(ENGINE_PATH);
    IExecutionContext* context = engine->createExecutionContext();

    // 2. 分配 GPU/CPU 内存
    void* d_input = nullptr, * d_output = nullptr;
    float* h_input = new float[1 * 3 * INPUT_SIZE * INPUT_SIZE];
    float* h_output = new float[OUTPUT_SIZE];

    CHECK_CUDA(cudaMalloc(&d_input, 1 * 3 * INPUT_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));

    // 3. 绑定输入输出
    void* bindings[] = { d_input, d_output };

    // 4. 读取图片
    Mat img = imread(IMAGE_PATH);
    if (img.empty()) { cerr << "Image not found!" << endl; return -1; }

    // 5. 预热
    cout << "Warming up TensorRT engine..." << endl;
    float scale;
    int top, left;
    preprocess(img, INPUT_SIZE, scale, top, left);

    for (int i = 0; i < 10; i++) {
        context->executeV2(bindings);
    }
    cudaDeviceSynchronize();
    cout << "Warmup done!\n" << endl;

    // 6. 测速
    vector<double> totalTimes, preTimes, inferTimes, postTimes;

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        auto t0 = chrono::high_resolution_clock::now();
        Mat processed = preprocess(img, INPUT_SIZE, scale, top, left);
        auto t1 = chrono::high_resolution_clock::now();

        // HWC -> CHW
        vector<Mat> channels(3);
        split(processed, channels);
        for (int c = 0; c < 3; c++) {
            memcpy(h_input + c * INPUT_SIZE * INPUT_SIZE, channels[c].data,
                INPUT_SIZE * INPUT_SIZE * sizeof(float));
        }
        CHECK_CUDA(cudaMemcpy(d_input, h_input,
            3 * INPUT_SIZE * INPUT_SIZE * sizeof(float),
            cudaMemcpyHostToDevice));
        auto t2 = chrono::high_resolution_clock::now();

        // 推理
        context->executeV2(bindings);
        cudaDeviceSynchronize();
        auto t3 = chrono::high_resolution_clock::now();

        CHECK_CUDA(cudaMemcpy(h_output, d_output,
            OUTPUT_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost));
        auto t4 = chrono::high_resolution_clock::now();

        auto dets = postprocess(h_output, img.rows, img.cols, INPUT_SIZE,
            CONF_THRESHOLD, IOU_THRESHOLD, scale, top, left);
        auto t5 = chrono::high_resolution_clock::now();

        double pre = chrono::duration<double, milli>(t1 - t0).count();
        double infer = chrono::duration<double, milli>(t3 - t2).count();
        double post = chrono::duration<double, milli>(t5 - t4).count();
        double total = chrono::duration<double, milli>(t5 - t0).count();

        preTimes.push_back(pre);
        inferTimes.push_back(infer);
        postTimes.push_back(post);
        totalTimes.push_back(total);
    }

    // 输出结果
    cout << "============================================================" << endl;
    cout << "          YOLOv5 TensorRT C++ 推理耗时统计" << endl;
    cout << "============================================================" << endl;
    cout << "预处理时间    : " << accumulate(preTimes.begin(), preTimes.end(), 0.0) / TEST_ITERATIONS << " ms" << endl;
    cout << "纯推理时间    : " << accumulate(inferTimes.begin(), inferTimes.end(), 0.0) / TEST_ITERATIONS << " ms" << endl;
    cout << "后处理时间    : " << accumulate(postTimes.begin(), postTimes.end(), 0.0) / TEST_ITERATIONS << " ms" << endl;
    cout << "端到端总耗时  : " << accumulate(totalTimes.begin(), totalTimes.end(), 0.0) / TEST_ITERATIONS << " ms" << endl;
    cout << "============================================================" << endl;

    // 可视化
    vector<string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    Mat finalImg = img.clone();
    auto dets = postprocess(h_output, img.rows, img.cols, INPUT_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, scale, top, left);
    for (auto& det : dets) {
        rectangle(finalImg, Point(det.x1, det.y1), Point(det.x2, det.y2), Scalar(0, 255, 0), 2);
        string label = classNames[det.classId] + " " + to_string(det.score).substr(0, 4);
        putText(finalImg, label, Point(det.x1, det.y1 - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }

    imwrite("result_trt_cpp.jpg", finalImg);
    imshow("TRT C++ Detection", finalImg);
    waitKey(0);

    // 释放资源
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}