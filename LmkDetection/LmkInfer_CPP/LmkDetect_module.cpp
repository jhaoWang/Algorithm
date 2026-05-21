#include "LmkDetect_module.h"
#include <windows.h>


// ======================= Onnx初始化类实现 ==========================
LmkDetection::LmkDetection()
	:is_loaded(false)
{
	m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LmkDetection");
	m_session_option = std::make_unique<Ort::SessionOptions>();
	m_session_option->SetIntraOpNumThreads(1);
	m_memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

LmkDetection::~LmkDetection() {};

bool LmkDetection::loadModel(const std::string& path)
{
	try {
		m_session = std::make_unique<Ort::Session>(*m_env, std::wstring(path.begin(), path.end()).c_str(), *m_session_option);

		Ort::AllocatorWithDefaultOptions allocator;
		m_inputNames.clear();
		m_inputNames_ptr.clear();
		size_t input_size = m_session->GetInputCount();
		for (size_t i = 0; i < input_size; i++)
		{
			auto name = m_session->GetInputNameAllocated(i, allocator);
			m_inputNames.emplace_back(name.get());
			std::cout << "Input" << i <<":" << name << std::endl;
		}

		for (auto& name : m_inputNames)
		{
			m_inputNames_ptr.push_back(name.c_str());
		}

		m_outputName.clear();
		m_outputName_ptr.clear();

		size_t output_size = m_session->GetOutputCount();
		for (size_t i = 0; i < output_size; i++)
		{
			auto name = m_session->GetOutputNameAllocated(i, allocator);
			m_outputName.emplace_back(name.get());
			std::cout << "Output" << i << ":" <<  name << std::endl;

			auto typeInfo = m_session->GetOutputTypeInfo(i);
			auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
			m_output_shape = tensorInfo.GetShape();
		}

		for (auto& name : m_outputName)
		{
			m_outputName_ptr.push_back(name.c_str());
		}

		is_loaded = true;
		std::cout << "ONNX load successfully!" << std::endl;
		return true;
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "ONNX load failed!" << e.what() << std::endl;
		return false;
	}
}

void LmkDetection::setInputName(const std::vector<std::string>& inputnames)
{
	if (inputnames.size() == 2)
	{
		m_inputNames = inputnames;
		m_inputNames_ptr.clear();
		for (auto name : m_inputNames)
		{
			m_inputNames_ptr.push_back(name.c_str());
		}
	}
}

bool LmkDetection::inference(const std::vector<float>& img, const std::vector<float>& mask, std::vector<float>& output)
{
	if (!is_loaded)
	{
		std::cerr << "Onnx Model has Not Load" << std::endl;
		return false;
	}

	try {
		std::vector<int64_t> img_shape{ 1, 3, 256, 256 };
		std::vector<int64_t> mask_shape{ 1, 1, 256, 256 };

		std::vector<Ort::Value> input_tensors;
		input_tensors.push_back(Ort::Value::CreateTensor(*m_memoryInfo, const_cast<float*>(img.data()), img.size(), img_shape.data(), img_shape.size()));
		input_tensors.push_back(Ort::Value::CreateTensor(*m_memoryInfo, const_cast<float*>(mask.data()), mask.size(), mask_shape.data(), mask_shape.size()));

		std::vector<Ort::Value> output_tensor = m_session->Run(
			Ort::RunOptions{ nullptr },
			m_inputNames_ptr.data(), input_tensors.data(), input_tensors.size(),
			m_outputName_ptr.data(), m_outputName_ptr.size()
		);

		float* output_ptr = output_tensor[0].GetTensorMutableData<float>();
		size_t output_size = output_tensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
		output.assign(output_ptr, output_ptr + output_size);

		return true;
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "Inference is failed" << e.what() <<  std::endl;
		return false;
	}
}

// ============================= 图像前后处理类实现 ===========================
ImageProcess::ImageProcess() {};

ImageProcess::~ImageProcess() {};

bool ImageProcess::loadImage(const std::string& path, cv::Mat& img, bool is_color)
{
	int flag = is_color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
	img = cv::imread(path, flag);

	if (img.empty())
	{
		std::cerr << "Failed to load img" << std::endl;
		return false;
	}
	return true;
}


std::vector<float> ImageProcess::preprocessImg(const cv::Mat& img)
{
	cv::Mat resizedImg, floatImg;
	cv::resize(img, resizedImg, cv::Size(256, 256));
	resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0f);
	cv::cvtColor(floatImg, floatImg, cv::COLOR_BGR2RGB);

	std::vector<float> input_data;
	input_data.reserve(1 * 3 * 256 * 256);

	for (int c = 0; c < 3; c++) {
		for (int h = 0; h < 256; h++) {
			for (int w = 0; w < 256; w++) {
				input_data.push_back(floatImg.at<cv::Vec3f>(h, w)[c]);
			}
		}
	}
	return input_data;
}

std::vector<float> ImageProcess::preprocessMask(const cv::Mat& mask)
{
	cv::Mat resizedMask, floatMask;
	cv::resize(mask, resizedMask, cv::Size(256, 256));
	resizedMask.convertTo(floatMask, CV_32FC1, 1.0 / 255.0f);

	std::vector<float> input_data;
	input_data.reserve(1 * 1 * 256 * 256);

	for (int h = 0; h < 256; h++) {
		for (int w = 0; w < 256; w++) {
			input_data.push_back(floatMask.at<float>(h, w));
		}
	}
	return input_data;
}

std::vector<float> ImageProcess::addMask(const std::vector<float>& img, const std::vector<float>& mask)
{
	std::vector<float> masked_img(img.size());
	for (int i = 0; i < masked_img.size(); i++)
	{
		int mask_index = i % (256 * 256);
		float m = mask[mask_index];
		masked_img[i] = img[i] * (1.0 - m) + m;
	}

	return masked_img;
}

cv::Mat ImageProcess::postprocess(const std::vector<float>& output, const preprocessParam& param)
{
	std::vector<cv::Point2f> landmarks(param.number_point);
	for (int i = 0; i < param.number_point; i++)
	{
		float x = output[2 * i];
		float y = output[2 * i + 1];

		x = (x >= param.input_size - 1) ? (param.input_size - 1) : x;
		y = (y >= param.input_size - 1) ? (param.input_size - 1) : y;

		landmarks[i] = cv::Point2f(x, y);
	}
	float scale_x = (float)param.input_size/ param.input_size;
	float scale_y = (float)param.input_size / param.input_size;

	cv::Mat black_background = cv::Mat::zeros(param.input_size, param.input_size, CV_8UC1);
	for (auto& p : landmarks) {
		int ori_x = cvRound(p.x * scale_x);
		int ori_y = cvRound(p.y * scale_y);
		cv::circle(black_background, cv::Point(ori_x, ori_y), 2, cv::Scalar(255), -1);
	}

	return black_background;
}

// ======================== Detection类实现 ========================
Detection::Detection()
{
	m_infer = std::make_unique<LmkDetection>();
	m_processor = std::make_unique<ImageProcess>();
}

Detection::~Detection() {};

bool Detection::Init(const std::string& model_path)
{
	bool success = m_infer->loadModel(model_path);
	if (success)
	//{
	//	std::vector<std::string> input_name{ "onnx::Mul_0", "onnx::Sub_1" };
	//	m_infer->setInputName(input_name);
	//}
	return success;
}

void Detection::setInputPath(const std::string& img_path, const std::string& mask_path)
{
	m_img_path = img_path;
	m_mask_path = mask_path;
}

bool Detection::run()
{
	if (!m_processor->loadImage(m_img_path, m_img, true)) return false;
	if (!m_processor->loadImage(m_mask_path, m_mask, false)) return false;

	std::vector<float> input_img = m_processor->preprocessImg(m_img);
	std::vector<float> input_mask = m_processor->preprocessMask(m_mask);

	std::vector<float> output_data;
	bool success = m_infer->inference(input_img, input_mask, output_data);
	if (!success) return false;

	cv::Mat result = m_processor->postprocess(output_data, m_param);
	showResult(result);
	return true;
}

void Detection::showResult(const cv::Mat& result)
{
	cv::Mat result_bgr;
	cv::cvtColor(result, result_bgr, cv::COLOR_RGB2BGR);
	cv::imshow("Inpainting result", result_bgr);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

