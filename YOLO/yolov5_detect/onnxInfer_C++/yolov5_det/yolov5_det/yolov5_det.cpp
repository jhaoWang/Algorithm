#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace std::chrono;

// ==================== 配置参数 ====================
const wchar_t* ONNX_MODEL_PATH = L"yolov5s.onnx";
const char* IMAGE_PATH = "bus.jpg";
const float CONF_THRESHOLD = 0.25f;
const float IOU_THRESHOLD = 0.45f;
const int INPUT_SIZE = 640;

// COCO 类别名称
vector<string> CLASS_NAMES = {
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

// ==================== 预处理 ====================
Mat preprocess(Mat& img, int input_size, double& scale, int& pad_w, int& pad_h) {
    int h = img.rows;
    int w = img.cols;
    scale = (double)input_size / max(h, w);
    int new_w = cvRound(w * scale);
    int new_h = cvRound(h * scale);
    resize(img, img, Size(new_w, new_h));

    pad_w = (input_size - new_w) / 2;
    pad_h = (input_size - new_h) / 2;
    Mat canvas;
    copyMakeBorder(img, canvas, pad_h, input_size - new_h - pad_h,
        pad_w, input_size - new_w - pad_w, BORDER_CONSTANT, Scalar(0, 0, 0));

    // BGR → RGB，和 Python 保持一致
    cvtColor(canvas, canvas, COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32F, 1 / 255.0);

    vector<Mat> channels(3);
    split(canvas, channels);

    Mat blob(1, 3 * input_size * input_size, CV_32F);
    float* data = blob.ptr<float>();
    for (int i = 0; i < 3; i++) {
        channels[i].copyTo(Mat(input_size, input_size, CV_32F, data + i * input_size * input_size));
    }
    return blob;
}

// ==================== 后处理（核心修复） ====================
struct DetResult {
    Rect box;
    float score;
    int class_id;
};

vector<DetResult> postprocess(float* preds, int ori_h, int ori_w, int input_size,
    float conf_thres, float iou_thres, double scale, int pad_w, int pad_h) {
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;

    for (int i = 0; i < 25200; i++) {
        float obj_conf = preds[i * 85 + 4];
        if (obj_conf < conf_thres) continue;

        float* class_scores = preds + i * 85 + 5;
        int class_id = max_element(class_scores, class_scores + 80) - class_scores;
        float score = obj_conf * class_scores[class_id];
        if (score < conf_thres) continue;

        // 模型输出：cx, cy, w, h（在 640x640 图像上的坐标）
        float cx = preds[i * 85 + 0];
        float cy = preds[i * 85 + 1];
        float w = preds[i * 85 + 2];
        float h = preds[i * 85 + 3];

        // 第一步：在 640x640 上转成 x1,y1,x2,y2
        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        float x2 = cx + w / 2;
        float y2 = cy + h / 2;

        // 第二步：减去填充，再除以缩放，还原到原图坐标
        x1 = (x1 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
        x2 = (x2 - pad_w) / scale;
        y2 = (y2 - pad_h) / scale;

        // 第三步：转成 NMS 需要的 Rect(x, y, w, h)
        int x = cvRound(x1);
        int y = cvRound(y1);
        int box_w = cvRound(x2 - x1);
        int box_h = cvRound(y2 - y1);

        boxes.emplace_back(x, y, box_w, box_h);
        confidences.push_back(score);
        classIds.push_back(class_id);
    }

    vector<int> indices;
    // 兼容所有 OpenCV 版本的 NMSBoxes 调用方式
    NMSBoxes(boxes, confidences, conf_thres, iou_thres, indices);

    vector<DetResult> results;
    for (int idx : indices) {
        results.push_back({ boxes[idx], confidences[idx], classIds[idx] });
    }
    return results;
}

// ==================== 主函数 ====================
int main() {
    Mat img = imread(IMAGE_PATH);
    int ori_h = img.rows;
    int ori_w = img.cols;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov5");
    Ort::SessionOptions session_options;
    Ort::Session session(env, ONNX_MODEL_PATH, session_options);

    auto total_start = high_resolution_clock::now();

    // 预处理
    auto t1 = high_resolution_clock::now();
    double scale;
    int pad_w, pad_h;
    Mat blob = preprocess(img, INPUT_SIZE, scale, pad_w, pad_h);
    auto t2 = high_resolution_clock::now();
    float pre_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0f;

    // 推理
    vector<int64_t> input_shape = { 1, 3, INPUT_SIZE, INPUT_SIZE };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), 3 * INPUT_SIZE * INPUT_SIZE,
        input_shape.data(), input_shape.size()
    );

    const char* input_names[] = { "images" };
    const char* output_names[] = { "output0" };

    t1 = high_resolution_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
    t2 = high_resolution_clock::now();
    float infer_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0f;

    // 后处理
    float* preds = output_tensors[0].GetTensorMutableData<float>();
    t1 = high_resolution_clock::now();
    auto results = postprocess(preds, ori_h, ori_w, INPUT_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, scale, pad_w, pad_h);
    t2 = high_resolution_clock::now();
    float post_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0f;

    // 总时间
    auto total_end = high_resolution_clock::now();
    float total_time = duration_cast<microseconds>(total_end - total_start).count() / 1000.0f;

    // 绘制（和 Python 保持一致）
    img = imread(IMAGE_PATH); // 重新读取原图，避免预处理修改
    for (auto& res : results) {
        rectangle(img, res.box, Scalar(0, 255, 0), 2);
        string label = CLASS_NAMES[res.class_id] + " " + to_string(res.score).substr(0, 4);
        putText(img, label, Point(res.box.x, res.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
    imwrite("result.jpg", img);
    imshow("result", img);
    waitKey(0);

    // 打印
    cout << "==================================================" << endl;
    cout << "预处理时间：\t" << pre_time << " ms" << endl;
    cout << "推理时间：\t" << infer_time << " ms" << endl;
    cout << "后处理时间：\t" << post_time << " ms" << endl;
    cout << "总耗时：\t" << total_time << " ms" << endl;
    cout << "检测目标数：\t" << results.size() << endl;
    cout << "==================================================" << endl;

    return 0;
}