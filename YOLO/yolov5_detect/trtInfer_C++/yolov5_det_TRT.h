#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

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

class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING)
		{
			std::cout << "[TRT][" << getSeverityStr(severity) << "] " << msg << std::endl;
		}
	}

private:
	static const char* getSeverityStr(Severity severity) {
		switch (severity) {
		case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
		case Severity::kERROR:          return "ERROR";
		case Severity::kWARNING:        return "WARNING";
		case Severity::kINFO:           return "INFO";
		case Severity::kVERBOSE:        return "VERBOSE";
		default:                        return "UNKNOWN";
		}
	}
};

class TrtInferEngine {
public:
	TrtInferEngine();
	TrtInferEngine(const TrtInferEngine&) = delete;
	TrtInferEngine& operator=(const TrtInferEngine&) = delete;
	TrtInferEngine(TrtInferEngine&&) = delete;
	TrtInferEngine& operator=(TrtInferEngine&&) = delete;

	~TrtInferEngine();

	bool loadEngine(const std::string& engine_path);
	bool createContext();
	bool allocateBuffer(const std::vector<size_t>& buffersizes);
	void* getBuffer(int index) const;
	nvinfer1::IExecutionContext* getContext() const;
	bool inference();
	void cleanUP();

private:
	Logger loger_;
	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	nvinfer1::IRuntime* runtime_;

	std::vector<void*> buffers_;
	std::vector<size_t> buffersSize_;

	bool isLoaded_;
	void freeBuffers();
};

class ImageProcess {
public:
	ImageProcess();
	~ImageProcess();

	bool loadImage(const std::string& path, cv::Mat& img);
	cv::Mat preprocessImg(const cv::Mat& img, int inputsize, double& scale, int& pad_w, int& pad_h);
	std::vector<DetResult> postProcessImg(const std::vector<float>& pred, int ori_h, int ori_w, int input_size,
		float conf_thres, float iou_thres, double scale, int pad_w, int pad_h);

	cv::Mat drawResult(const std::vector<DetResult>& res_post, const std::vector<std::string>& class_det, cv::Mat& result);
};

class yoloDet
{
public:
	yoloDet();
	~yoloDet();
	bool Init(const std::string& modelpath);
	bool loadImg(const std::string& img_path);
	bool runInference();
	void DrawandshowResult();
private:
	std::unique_ptr<TrtInferEngine> Infer_;
	std::unique_ptr<ImageProcess> Processor_;

	int h_;
	int w_;
	double scale;
	int pad_w, pad_h;
	bool isInitialized;
	cv::Mat m_img;
	cv::Mat input_img;
	cv::Mat out_img;
	std::vector<DetResult> det_res_;

	bool copyToDevice();
	bool copyFromDevice();
};