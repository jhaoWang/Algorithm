#include "yolov5Det_moudle.h"
#include <algorithm>
#include <chrono> 

// ======================= Onnx相关类实现 ==========================
Yolov5OnnxEngine::Yolov5OnnxEngine()
	: is_loaded(false)
{
	m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Yolov5OnnxInference");
	m_session_option = std::make_unique<Ort::SessionOptions>();
	m_session_option->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	//m_session_option->SetIntraOpNumThreads(std::thread::hardware_concurrency());
	//m_session_option->SetIntraOpNumThreads(1);
	m_memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

Yolov5OnnxEngine::~Yolov5OnnxEngine(){}

bool Yolov5OnnxEngine::loadModel(const std::string& path)
{
	try {
		m_session = std::make_unique<Ort::Session>(*m_env, std::wstring(path.begin(), path.end()).c_str(), *m_session_option);
		Ort::AllocatorWithDefaultOptions allocator;

		size_t input_size = m_session->GetInputCount();
		m_inputNames.clear();
		m_inputNames_ptr.clear();
		for (size_t i = 0; i < input_size; i++)
		{
			auto name = m_session->GetInputNameAllocated(i, allocator);
			m_inputNames.emplace_back(name.get());
			std::cout << "Input" << i << ":" << name << std::endl;
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
			std::cout << "Output" << i << ":" << name << std::endl;

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

void Yolov5OnnxEngine::setInputName(const std::vector<std::string>& inputnames)
{
	if (inputnames.size() == 1)
	{
		m_inputNames = inputnames;
		m_inputNames_ptr.clear();
		for (auto& name : m_inputNames)
		{
			m_inputNames_ptr.push_back(name.c_str());
		}
	}
}

bool Yolov5OnnxEngine::inference(const cv::Mat& img, std::vector<float>& output)
{
	if (!is_loaded)
	{
		std::cerr << "Onnx model has not load!" << std::endl;
		return false;
	}

	try {
		std::vector<int64_t> input_shape{ 1, 3, 640, 640 };

		std::vector<Ort::Value> input_tensor;
		input_tensor.push_back(Ort::Value::CreateTensor(*m_memoryInfo, (float*)img.data, 3 * 640 * 640, input_shape.data(), input_shape.size()));

		std::vector<Ort::Value> output_tensor = m_session->Run(
			Ort::RunOptions{ nullptr },
			m_inputNames_ptr.data(), input_tensor.data(), 1,
			m_outputName_ptr.data(), 1
		);

		float* output_ptr = output_tensor[0].GetTensorMutableData<float>();
		size_t output_size = output_tensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
		output.assign(output_ptr, output_ptr + output_size);

		return true;
	}
	catch (const Ort::Exception& e) {
		std::cerr << "Inference is failed" << e.what() << std::endl;
		return false;
	}
}

// ======================= 图像处理类实现 ==========================
ImageProcess::ImageProcess() {};

ImageProcess::~ImageProcess() {};

bool ImageProcess::loadImage(const std::string& path, cv::Mat& img)
{
	img = cv::imread(path, cv::IMREAD_COLOR);
	if (img.empty())
	{
		std::cerr << "Failed to load img" << std::endl;
		return false;
	}
	return true;
}

cv::Mat ImageProcess::preprocessImg(const cv::Mat& img, int input_size, double& scale, int& pad_w, int& pad_h)
{
	int h = img.rows;
	int w = img.cols;

	scale = (double)input_size / std::max(h, w);
	int new_w = cvRound(w * scale);
	int new_h = cvRound(h * scale);
	
	cv::Mat img_resized;
	cv::resize(img, img_resized, cv::Size(new_w, new_h));

	pad_w = (input_size - new_w) / 2;
	pad_h = (input_size - new_h) / 2;

	cv::Mat canvas;
	cv::copyMakeBorder(img_resized, canvas, pad_h, input_size - new_h - pad_h,
		pad_w, input_size - new_w - pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
	canvas.convertTo(canvas, CV_32F, 1 / 255.0);

	//std::vector<float> input;
	//input.reserve(3 * input_size * input_size);

	//for (int c = 0; c < 3; c++) {
	//	for (int y = 0; y < input_size; y++) {
	//		for (int x = 0; x < input_size; x++) {
	//			input.push_back(canvas.at<cv::Vec3f>(y, x)[c]);
	//		}
	//	}
	//}
	std::vector<cv::Mat> channels(3);
	cv::split(canvas, channels);
	cv::Mat blob(1, 3 * input_size * input_size, CV_32F);
	// std::vector<float> input(3 * input_size * input_size);
	float* data = blob.ptr<float>();
	//float* data = input.data();
	for (int i = 0; i < 3; i++) {
		channels[i].copyTo(cv::Mat(input_size, input_size, CV_32F, data + i * input_size * input_size));
	}

	return blob;
}

std::vector<DetResult> ImageProcess::postProcessImg(const std::vector<float>& preds, int ori_h, int ori_w, int input_size,
	float conf_thres, float iou_thres, double scale, int pad_w, int pad_h)
{
	std::vector<cv::Rect> boxes;
	std::vector<float> confidences;
	std::vector<int> classIds;

	for (int i = 0; i < 25200; i++)
	{
		float obj_conf = preds[i * 85 + 4];
		if (obj_conf < conf_thres) continue;

		const float* class_scores = preds.data() + i * 85 + 5;
		int class_id = std::max_element(class_scores, class_scores + 80) - class_scores;
		float score = obj_conf * class_scores[class_id];
		if (score < conf_thres) continue;

		float cx = preds[i * 85 + 0];
		float cy = preds[i * 85 + 1];
		float w = preds[i * 85 + 2];
		float h = preds[i * 85 + 3];

		float x1 = cx - w / 2;
		float y1 = cy - h / 2;
		float x2 = cx + w / 2;
		float y2 = cy + h / 2;

		x1 = (x1 - pad_w) / scale;
		y1 = (y1 - pad_h) / scale;
		x2 = (x2 - pad_w) / scale;
		y2 = (y2 - pad_h) / scale;

		int x = cvRound(x1);
		int y = cvRound(y1);
		int box_w = cvRound(x2 - x1);
		int box_h = cvRound(y2 - y1);

		boxes.emplace_back(x, y, box_w, box_h);
		confidences.push_back(score);
		classIds.push_back(class_id);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_thres, iou_thres, indices);

	std::vector<DetResult> results;
	for (int idx : indices) {
		results.push_back({ boxes[idx], confidences[idx], classIds[idx] });
	}
	return results;
}

cv::Mat ImageProcess::drawResult(const std::vector<DetResult>& res_post, const std::vector<std::string>& det_class, cv::Mat& result)
{
	for (auto& res : res_post)
	{
		cv::rectangle(result, res.box, cv::Scalar(0, 255, 0), 2);
		std::string label = det_class[res.class_id] + " " + std::to_string(res.score).substr(0, 4);
		cv::putText(result, label, cv::Point(res.box.x, res.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
	}
	return result;
}

// ======================= 实际检测类实现 ==========================

Detection::Detection()
{
	m_infer = std::make_unique<Yolov5OnnxEngine>();
	m_processor = std::make_unique<ImageProcess>();
}

Detection::~Detection() {};

bool Detection::Init(const std::string& path)
{
	bool success = m_infer->loadModel(path);
	if (success) return success;
}

void Detection::setInputPath(const std::string& img_path)
{
	m_img_path = img_path;
}

bool Detection::run()
{
	auto total_start = std::chrono::high_resolution_clock::now();

	if (!m_processor->loadImage(m_img_path, m_img)) return false;

	int ori_h = m_img.rows;
	int ori_w = m_img.cols;

	double scale;
	int pad_w, pad_h;

	auto pre_start = std::chrono::high_resolution_clock::now();
	cv::Mat input_img = m_processor->preprocessImg(m_img, 640, scale, pad_w, pad_h);
	auto pre_end = std::chrono::high_resolution_clock::now();
	double pre_time = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();

	std::vector<float> output_data;
	auto infer_start = std::chrono::high_resolution_clock::now();
	bool success = m_infer->inference(input_img, output_data);
	auto infer_end = std::chrono::high_resolution_clock::now();
	double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
	if (!success) return false;

	auto post_start = std::chrono::high_resolution_clock::now();
	std::vector<DetResult> det_res = m_processor->postProcessImg(output_data, ori_h, ori_w, 640, 0.25f, 0.45f, scale, pad_w, pad_h);
	auto post_end = std::chrono::high_resolution_clock::now();
	double post_time = std::chrono::duration<double, std::milli>(post_end - post_start).count();

	auto total_end = std::chrono::high_resolution_clock::now();
	double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

	printf("=========================================\n");
	printf("预处理时间: %.2f ms\n", pre_time);
	printf("推理时间  : %.2f ms\n", infer_time);
	printf("后处理时间: %.2f ms\n", post_time);
	printf("总耗时    : %.2f ms\n", total_time);
	printf("=========================================\n");

	cv::Mat after_pos;
	res_final = m_processor->drawResult(det_res, CLASS_NAMES, m_img);
	showResult(res_final);

	return true;
}

void Detection::showResult(const cv::Mat& result)
{
	cv::imshow("Inpainting result", result);
	cv::waitKey(0);
	cv::destroyAllWindows();
}