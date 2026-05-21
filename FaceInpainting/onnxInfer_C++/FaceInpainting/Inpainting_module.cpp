#include "Inpainting_module.h"
#include <algorithm>
#include <windows.h>

// ====================== ONNXInferenceEngine 实现 ======================
OnnxInferenceEngine::OnnxInferenceEngine()
	:m_is_loaded(false)
{
	m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "InpaintEngine");
	m_session_options = std::make_unique<Ort::SessionOptions>();
	m_session_options->SetIntraOpNumThreads(1);
	m_memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

OnnxInferenceEngine::~OnnxInferenceEngine() {};

bool OnnxInferenceEngine::loadModel(const std::string& path)
{
	try {
		m_session = std::make_unique<Ort::Session>(*m_env, std::wstring(path.begin(), path.end()).c_str(), *m_session_options);

		Ort::AllocatorWithDefaultOptions allocater;

		m_input_name.clear();
		m_input_name_ptr.clear();
		size_t size_input = m_session->GetInputCount();
		for (size_t i = 0; i < size_input; i++)
		{
			auto name = m_session->GetInputNameAllocated(i, allocater);
			m_input_name.emplace_back(name.get());
			std::cout << "input" << i << ":" << name.get() << std::endl;
		}

		for (auto& name : m_input_name)
		{
			m_input_name_ptr.push_back(name.c_str());
		}

		m_output_name.clear();
		m_output_name_ptr.clear();

		size_t size_output = m_session->GetOutputCount();
		for (size_t i = 0; i < size_output; i++)
		{
			auto name = m_session->GetOutputNameAllocated(i, allocater);
			m_output_name.emplace_back(name.get());
			std::cout << "output" << i << ":" << name.get() << std::endl;
			
			auto typeInfo = m_session->GetOutputTypeInfo(i);
			auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
			m_output_shape = tensorInfo.GetShape();
		}

		for (auto& name : m_output_name)
		{
			m_output_name_ptr.push_back(name.c_str());
		}

		m_is_loaded = true;
		std::cout << "ONNX Model load success!" << std::endl;
		return true;
	}
	catch (const Ort::Exception& e){
		std::cout << "Failed to load model" << e.what() << std::endl;
		return false;
	}
}

void OnnxInferenceEngine::setInputName(const std::vector<std::string>& inputs)
{
	if (inputs.size() == 3)
	{
		m_input_name = inputs;
		m_input_name_ptr.clear();
		for (const auto& name : m_input_name)
		{
			m_input_name_ptr.push_back(name.c_str());
		}
	}
}

bool OnnxInferenceEngine::inference(const std::vector<float>& img
	, const std::vector<float>& lmk
	, const std::vector<float>& mask
	,std::vector<float>& output)
{
	if (!m_is_loaded)
	{
		std::cerr << "Model not loaded" << std::endl;
		return false;
	}

	try {
		std::vector<int64_t> img_shape = { 1, 3, 256, 256 };
		std::vector<int64_t> lmk_shape = { 1, 1, 256, 256 };
		std::vector<int64_t> mask_shape = { 1, 1, 256, 256 };

		std::vector<Ort::Value> input_tensors;
		input_tensors.push_back(Ort::Value::CreateTensor(*m_memoryInfo, const_cast<float*>(img.data()), img.size(), img_shape.data(), img_shape.size()));
		input_tensors.push_back(Ort::Value::CreateTensor(*m_memoryInfo, const_cast<float*>(lmk.data()), lmk.size(), lmk_shape.data(), lmk_shape.size()));
		input_tensors.push_back(Ort::Value::CreateTensor(*m_memoryInfo, const_cast<float*>(mask.data()), mask.size(), mask_shape.data(), mask_shape.size()));

		std::vector<Ort::Value> output_tensor = m_session->Run(
			Ort::RunOptions{ nullptr },
			m_input_name_ptr.data(), input_tensors.data(), input_tensors.size(),
			m_output_name_ptr.data(), m_output_name_ptr.size()
		);

		float* output_ptr = output_tensor[0].GetTensorMutableData<float>();
		size_t output_size = output_tensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
		output.assign(output_ptr, output_ptr + output_size);

		return true;
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "Inference failed" << e.what() << std::endl;
		return false;
	}
}

// ====================== ImageProcess 实现 ======================
ImageProcess::ImageProcess() {};

ImageProcess::~ImageProcess() {};

bool ImageProcess::loadImage(const std::string& path, cv::Mat& img, bool is_color)
{
	int flag = is_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
	img = cv::imread(path, flag);
	if (img.empty())
	{
		std::cerr << "Failed to load img" << path << std::endl;
		return false;
	}
	return true;
}


std::vector<float> ImageProcess::convertHWCtoCHW(const cv::Mat& img, int channel)
{
	int w = img.cols;
	int h = img.rows;
	int c = channel;

	std::vector<float> result(w * h * c);

	if (c == 3)
	{
		std::vector<cv::Mat> splitedImg(3);
		cv::split(img, splitedImg);
		for (int i = 0; i < c; i++)
		{
			memcpy(result.data() + i * h * w, splitedImg[i].data, h * w * sizeof(float));
		}
	}
	else
	{
		memcpy(result.data(), img.data, h * w * sizeof(float));
	}
	return result;
}

std::vector<float> ImageProcess::preprocessImage(const cv::Mat& img, const PreprocessParam& param)
{
	cv::Mat resizedImg;
	cv::resize(img, resizedImg, cv::Size(param.w_target, param.h_target));

	cv::Mat rgbImg;
	cv::cvtColor(resizedImg, rgbImg, cv::COLOR_BGR2RGB);

	std::vector<float> input_img(1 * 3 * 256 * 256);

	for (int h = 0; h < 256; h++) {
		for (int w = 0; w < 256; w++) {
			cv::Vec3b pix = rgbImg.at<cv::Vec3b>(h, w);
			input_img[0 * 256 * 256 + h * 256 + w] = pix[0] / 127.5f - 1.0f;
			input_img[1 * 256 * 256 + h * 256 + w] = pix[1] / 127.5f - 1.0f;
			input_img[2 * 256 * 256 + h * 256 + w] = pix[2] / 127.5f - 1.0f;
		}
	}

	return input_img;
}

std::vector<float> ImageProcess::preprocessMask(const cv::Mat& mask, const PreprocessParam& param)
{
	cv::Mat resizedMask;
	cv::resize(mask, resizedMask, cv::Size(param.w_target, param.h_target));

	cv::threshold(resizedMask, resizedMask, 0, 255, cv::THRESH_BINARY);

	std::vector<float> input_mask(1 * 1 * 256 * 256);

	for (int h = 0; h < 256; h++)
		for (int w = 0; w < 256; w++)
			input_mask[h * 256 + w] = resizedMask.at<uchar>(h, w) / 255.0f;

	return input_mask;
}

std::vector<float> ImageProcess::preprocessLmk(const cv::Mat& lmk, const PreprocessParam& param)
{
	cv::Mat resizedLmk;
	cv::resize(lmk, resizedLmk, cv::Size(param.w_target, param.h_target));

	std::vector<float> input_lmk(1 * 1 * 256 * 256);

	for (int h = 0; h < 256; h++)
		for (int w = 0; w < 256; w++)
			input_lmk[h * 256 + w] = resizedLmk.at<uchar>(h, w) / 255.0f;

	return input_lmk;
}

cv::Mat ImageProcess::postprocessImage(const std::vector<float>& output, const std::vector<int64_t>& output_shape)
{
	int batch = output_shape[0];
	int channels = output_shape[1];
	int height = output_shape[2];
	int width = output_shape[3];

	cv::Mat result(height, width, CV_32FC3);

	// 转换 CHW 到 HWC
	for (int c = 0; c < channels; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				float value = output[c * height * width + h * width + w];
				result.at<cv::Vec3f>(h, w)[c] = value;
			}
		}
	}
	result *= 255.0f;
	cv::threshold(result, result, 255.0f, 255.0f, cv::THRESH_TRUNC);
	cv::threshold(result, result, 0.0f, 0.0f, cv::THRESH_TOZERO);
	cv::Mat result_8u;
	result.convertTo(result_8u, CV_8UC3);
	return result_8u;
}

// ====================== Inpainting 实现 ======================
Inpainting::Inpainting() {
	m_inferenceEngine = std::make_unique<OnnxInferenceEngine>();
	m_processor = std::make_unique<ImageProcess>();
}

Inpainting::~Inpainting() {};

bool Inpainting::Init(const std::string& model_path)
{
	bool success = m_inferenceEngine->loadModel(model_path);
	if (success)
	{
		std::vector<std::string> input_names{ "onnx::Mul_0", "onnx::Concat_1", "masks" };
		m_inferenceEngine->setInputName(input_names);
	}
	return success;
}

void Inpainting::setInputPath(const std::string& img_path, const std::string& lmk_path, const std::string& mask_path)
{
	m_img_path = img_path;
	m_lmk_path = lmk_path;
	m_mask_path = mask_path;
}

bool Inpainting::run()
{
	// 1.加载图片
	if (!m_processor->loadImage(m_img_path, m_img, true)) return false;
	if (!m_processor->loadImage(m_lmk_path, m_lmk, false)) return false;
	if (!m_processor->loadImage(m_mask_path, m_mask, false)) return false;


	// 2.输入数据预处理
	std::vector<float> img_data = m_processor->preprocessImage(m_img, m_params);
	std::vector<float> lmk_data = m_processor->preprocessMask(m_lmk, m_params);
	std::vector<float> mask_data = m_processor->preprocessMask(m_mask, m_params);

	// 3.推理
	std::vector<float> output_data;
	bool success = m_inferenceEngine->inference(img_data, lmk_data, mask_data, output_data);
	if (!success) return false;

	// 4.后处理
	std::vector<int64_t> output_shape = m_inferenceEngine->getOutputShape();
	cv::Mat result = m_processor->postprocessImage(output_data, output_shape);

	showResult(result);
}

void Inpainting::showResult(const cv::Mat& result)
{
	cv::Mat result_bgr;
	cv::cvtColor(result, result_bgr, cv::COLOR_RGB2BGR);
	cv::imshow("Inpainting result", result_bgr);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

bool Inpainting::saveResult(const cv::Mat& result, const std::string& path)
{
	cv::Mat result_bgr;
	cv::cvtColor(result, result_bgr, cv::COLOR_RGB2BGR);
	cv::imwrite(path, result_bgr);
}