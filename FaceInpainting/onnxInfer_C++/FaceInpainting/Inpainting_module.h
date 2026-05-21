#pragma once
#include <memory>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2\opencv.hpp>



// 图像预处理参数结构体
struct PreprocessParam {
	int w_target = 256;
	int h_target = 256;
	float imgScale_min = -1.0f;
	float imgScale_max = 1.0f;
	float maskScale_min = 0.0f;
	float maskScale_max = 1.0f;
};

// ONNX加载初始化类
class OnnxInferenceEngine {
public:
	OnnxInferenceEngine();
	~OnnxInferenceEngine();
	bool loadModel(const std::string& path);
	void setInputName(const std::vector<std::string>& inputnames);
	bool inference(const std::vector<float>& imgdata, const std::vector<float>& lmkdata, const std::vector<float>& maskdata, std::vector<float>& outputdata);
	std::vector<int64_t> getOutputShape() const { return m_output_shape; }

private:
	std::unique_ptr<Ort::Env> m_env;
	std::unique_ptr<Ort::Session> m_session;
	std::unique_ptr<Ort::SessionOptions> m_session_options;
	std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;

	std::vector<std::string> m_input_name;
	std::vector<const char*> m_input_name_ptr;
	std::vector<std::string> m_output_name;
	std::vector<const char*> m_output_name_ptr;

	std::vector<int64_t> m_output_shape;
	bool m_is_loaded;
};

// 图像处理类
class ImageProcess {
public:
	ImageProcess();
	~ImageProcess();

	bool loadImage(const std::string& path, cv::Mat& img, bool is_color);
	std::vector<float> preprocessImage(const cv::Mat& img, const PreprocessParam& param);
	std::vector<float> preprocessMask(const cv::Mat& mask, const PreprocessParam& param);
	std::vector<float> preprocessLmk(const cv::Mat& lmk, const PreprocessParam& param);
	cv::Mat postprocessImage(const std::vector<float>& output_data, const std::vector<int64_t>& output_shape);

private:
	std::vector<float> convertHWCtoCHW(const cv::Mat& img, int channel);
};

class Inpainting {
public:
	Inpainting();
	~Inpainting();
	bool Init(const std::string& model_path);
	void setInputPath(const std::string& img_path, const std::string& lmk_path, const std::string& mask_path);
	bool run();
	void showResult(const cv::Mat& result);
	bool saveResult(const cv::Mat& result, const std::string& path);

private:
	std::unique_ptr<OnnxInferenceEngine> m_inferenceEngine;
	std::unique_ptr<ImageProcess> m_processor;
	PreprocessParam m_params;

	std::string m_img_path;
	std::string m_mask_path;
	std::string m_lmk_path;

	cv::Mat m_img;
	cv::Mat m_lmk;
	cv::Mat m_mask;
};
