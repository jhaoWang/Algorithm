#pragma once

#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2\opencv.hpp>

static std::vector<std::string> CLASS_NAMES = {
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

struct DetResult {
	cv::Rect box;
	float score;
	int class_id;
};

class Yolov5OnnxEngine {
public:
	Yolov5OnnxEngine();
	~Yolov5OnnxEngine();
	bool loadModel(const std::string& model_path);
	void setInputName(const std::vector<std::string>& inputname);
	bool inference(const std::vector<float>& img, std::vector<float>& output);
	std::vector<int64_t> getOutputShape() const { return m_output_shape; };

private:
	std::unique_ptr<Ort::Env> m_env;
	std::unique_ptr<Ort::Session> m_session;
	std::unique_ptr<Ort::SessionOptions> m_session_option;
	std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;

	std::vector<std::string> m_inputNames;
	std::vector<const char*> m_inputNames_ptr;
	std::vector<std::string> m_outputName;
	std::vector<const char*> m_outputName_ptr;

	std::vector<int64_t> m_output_shape;
	bool is_loaded;
};

class ImageProcess {
public:
	ImageProcess();
	~ImageProcess();

	bool loadImage(const std::string& path, cv::Mat& img);
	std::vector<float> preprocessImg(const cv::Mat& img, int inputsize, double& scale, int& pad_w, int& pad_h);
	std::vector<DetResult> postProcessImg(const std::vector<float>& pred, int ori_h, int ori_w, int input_size,
		float conf_thres, float iou_thres, double scale, int pad_w, int pad_h);

	cv::Mat drawResult(const std::vector<DetResult>& res_post, const std::vector<std::string>& class_det, cv::Mat& result);
};

class Detection {
public:
	Detection();
	~Detection();
	bool Init(const std::string& model_path);
	void setInputPath(const std::string& img_path);
	bool run();
	void showResult(const cv::Mat& result);

private:
	std::unique_ptr<Yolov5OnnxEngine> m_infer;
	std::unique_ptr<ImageProcess> m_processor;

	std::string m_img_path;

	cv::Mat m_img;
	cv::Mat res_final;
};