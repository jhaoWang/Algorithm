#include "yolov5_det_TRT.h"
#include <fstream>
#include <chrono>
using namespace std::chrono;

// ==================== TrtInferEngine类实现 =======================
TrtInferEngine::TrtInferEngine()
	: engine_(nullptr)
	, context_(nullptr)
	, runtime_(nullptr)
	, isLoaded_(false)
{ }

TrtInferEngine::~TrtInferEngine()
{
	cleanUP();
}

bool TrtInferEngine::loadEngine(const std::string& path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		std::cerr << "ERROR: cannot open the engine file" + path << std::endl;
		return false;
	}

	file.seekg(0, std::ios::end);
	size_t file_size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> data(file_size);
	file.read(data.data(), file_size);
	file.close();

	runtime_ = nvinfer1::createInferRuntime(loger_);
	if (!runtime_)
	{
		std::cerr << "ERROR: Create runtime failed" << std::endl;
		return false;
	}

	engine_ = runtime_->deserializeCudaEngine(data.data(), file_size);
	if (!engine_)
	{
		std::cerr << "ERROR: Deserialized engine failed" << std::endl;
		return false;
	}

	isLoaded_ = true;
	std::cout << "Engine successfully loaded" << std::endl;
	return true;
}

bool TrtInferEngine::createContext()
{
	if (!engine_)
	{
		std::cerr << "ERROR: Engine not loaded" << std::endl;
		return false;
	}

	context_ = engine_->createExecutionContext();
	if (!context_)
	{
		std::cerr << "ERROR: Context create failed" << std::endl;
		return false;
	}

	std::cout << "Context successfully created" << std::endl;
	return true;
}

bool TrtInferEngine::allocateBuffer(const std::vector<size_t>& buffersizes)
{
	if (!context_)
	{
		std::cerr << "ERROR: Context is not ready" << std::endl;
		return false;
	}

	freeBuffers();
	buffersSize_ = buffersizes;

	for (size_t i = 0; i < buffersizes.size(); i++)
	{
		void* buffer = nullptr;
		cudaError err = cudaMalloc(&buffer, buffersizes[i]);
		if (err != cudaSuccess)
		{
			std::cerr << "ERROR: allocate CUDA memory failed for buffer " << i
				<< ", size: " << buffersizes[i]
				<< ", error: " << cudaGetErrorString(err) << std::endl;

			freeBuffers();
			return false;
		}
		buffers_.push_back(buffer);
	}

	std::cout << "Allocated " << buffers_.size() << " CUDA buffers successfully" << std::endl;
	return true;
}

bool TrtInferEngine::inference()
{
	if (!context_ || buffers_.empty())
	{
		std::cerr << "ERROR: Context or buffer are not ready" << std::endl;
		return false;
	}

	bool success = context_->executeV2(buffers_.data());
	if (!success) {
		std::cerr << "ERROR: inference executeV2 failed" << std::endl;
	}
	return success;
}

void TrtInferEngine::cleanUP()
{
	freeBuffers();

	if (context_)
	{
		delete context_;
		context_ = nullptr;
	}

	if (engine_)
	{
		delete engine_;
		engine_ = nullptr;
	}

	if (runtime_)
	{
		delete runtime_;
		runtime_ = nullptr;
	}

	isLoaded_ = false;
	std::cout << "TrtInferEngine resources released successfully" << std::endl;
}

void* TrtInferEngine::getBuffer(int index) const
{
	if (index < 0 || (size_t)index >= buffers_.size())
	{
		std::cerr << "ERROR: buffer index out of range" << std::endl;
		return nullptr;
	}
	return buffers_[index];
}

nvinfer1::IExecutionContext* TrtInferEngine::getContext() const
{
	if (!context_)
	{
		std::cerr << "ERROR: context is null" << std::endl;
	}
	return context_;
}

void TrtInferEngine::freeBuffers() {
	for (void* buffer : buffers_) {
		if (buffer) {
			cudaFree(buffer);
		}
	}
	buffers_.clear();
	buffersSize_.clear();
}

// ======================= IMGProcess类实现 ==========================
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

// ========================== 检测类实现 ============================
yoloDet::yoloDet()
	: h_(640)
	, w_(640)
	, isInitialized(false)
{
	Infer_ = std::make_unique<TrtInferEngine>();
	Processor_ = std::make_unique<ImageProcess>();
}

yoloDet::~yoloDet()
{
	Infer_->cleanUP();
}

bool yoloDet::Init(const std::string& modelpath)
{
	if (!Infer_->loadEngine(modelpath)) return false;
	if (!Infer_->createContext()) return false;

	size_t img_size = 3 * h_ * w_ * sizeof(float);
	size_t out_size = 1 * 25200 * 85 * sizeof(float);

	std::vector<size_t> bufferSizes{ img_size, out_size };

	if (!Infer_->allocateBuffer(bufferSizes)) return false;

	for (int i = 0; i < 10; ++i)
	{
		Infer_->inference();
		cudaDeviceSynchronize();
	}

	isInitialized = true;
	std::cout << "yoloDet initialized successfully" << std::endl;
	return true;
}

bool yoloDet::loadImg(const std::string& img_path)
{
	if (!Processor_->loadImage(img_path, m_img)) return false;
	return true;
}

bool yoloDet::runInference()
{
	if (!isInitialized || m_img.empty())
	{
		std::cerr << "ERROR: Not inited or Not load imgs" << std::endl;
		return false;
	}

	auto total_start = high_resolution_clock::now();
	auto pre_start = high_resolution_clock::now();
	if (!copyToDevice())
	{
		std::cerr << "ERROR: Copy to device failed" << std::endl;
		return false;
	}
	auto pre_end = high_resolution_clock::now();
	double pre_time = duration<double, std::milli>(pre_end - pre_start).count();

	auto infer_start = high_resolution_clock::now();
	if (!Infer_->inference())
	{
		std::cerr << "ERROR: Failed to infer" << std::endl;
		return false;
	}
	cudaDeviceSynchronize();
	auto infer_end = high_resolution_clock::now();
	double infer_time = duration<double, std::milli>(infer_end - infer_start).count();

	auto post_start = high_resolution_clock::now();
	if (!copyFromDevice())
	{
		std::cerr << "ERROR: Copy from device failed" << std::endl;
		return false;
	}
	auto post_end = high_resolution_clock::now();
	double post_time = duration<double, std::milli>(post_end - post_start).count();

	auto total_end = high_resolution_clock::now();
	double total_time = duration<double, std::milli>(total_end - total_start).count();

	std::cout << "\n==================== times ====================\n";
	std::cout << "pre: " << pre_time << " ms\n";
	std::cout << "infer:     " << infer_time << " ms\n";
	std::cout << "post: " << post_time << " ms\n";
	std::cout << "==================================================\n";
	std::cout << "total:  " << total_time << " ms\n";
	std::cout << "==================================================\n\n";

	std::cout << "Inference completed successfully" << std::endl;
	return true;
}

bool yoloDet::copyToDevice()
{
	if (m_img.empty() || !isInitialized)
	{
		std::cerr << "ERROR: Not initialized or images not loaded" << std::endl;
		return false;
	}

	int ori_h = m_img.rows;
	int ori_w = m_img.cols;

	input_img = Processor_->preprocessImg(m_img, 640, scale, pad_w, pad_h);

	cudaError_t err;
	err = cudaMemcpy(Infer_->getBuffer(0), input_img.data, input_img.total() * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) return false;

	return true;
}

bool yoloDet::copyFromDevice()
{
	size_t outSize = 1 * 25200 * 85 * sizeof(float);
	std::vector<float> outData(outSize / sizeof(float));
	cudaError err = cudaMemcpy(outData.data(), Infer_->getBuffer(1), outSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) return false;

	det_res_ = Processor_->postProcessImg(outData, m_img.rows, m_img.cols, 640, 0.25f, 0.45f, scale, pad_w, pad_h);
	return true;
}

void yoloDet::DrawandshowResult()
{
	out_img = Processor_->drawResult(det_res_, CLASS_NAMES, m_img);
	cv::imshow("Inpainting result", out_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}