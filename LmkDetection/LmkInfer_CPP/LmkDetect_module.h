#pragma once
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2\opencv.hpp>

struct preprocessParam {
	int input_size = 256;
	int number_point = 68;
};

class LmkDetection {
public:
	LmkDetection();
	~LmkDetection();
	bool loadModel(const std::string& model_path);
	void setInputName(const std::vector<std::string>& inputname);
	bool inference(const std::vector<float>& img, const std::vector<float>& mask, std::vector<float>& output);
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

	bool loadImage(const std::string& img_path, cv::Mat& img, bool is_color);
	std::vector<float> preprocessImg(const cv::Mat& img);
	std::vector<float> preprocessMask(const cv::Mat& mask);
	std::vector<float> addMask(const std::vector<float>& img, const std::vector<float>& mask);
	cv::Mat postprocess(const std::vector<float>& output, const preprocessParam& param);

};

class Detection {
public:
	Detection();
	~Detection();
	bool Init(const std::string& model_path);
	void setInputPath(const std::string& img_path, const std::string& mask_path);
	bool run();
	void showResult(const cv::Mat& result);
	bool saveResult(const cv::Mat& result, const std::string& path);

private:
	std::unique_ptr<LmkDetection> m_infer;
	std::unique_ptr<ImageProcess> m_processor;
	preprocessParam m_param;

	std::string m_img_path;
	std::string m_mask_path;

	cv::Mat m_img;
	cv::Mat m_mask;
};