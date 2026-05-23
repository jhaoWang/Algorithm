#include "InpaintingTrT_module.h"
#include <fstream>
#include <chrono>
using namespace std::chrono;


// =================== TRT准备类实现 =======================
TRTInferEngine::TRTInferEngine()
	: engine_(nullptr)
	, context_(nullptr)
	, runtime_(nullptr)
	, isLoaded_(false)
{
}

TRTInferEngine::~TRTInferEngine()
{

}

bool TRTInferEngine::loadEngine(const std::string& engine_path)
{
	std::ifstream file(engine_path, std::ios::binary);
	if (!file)
	{
		std::cerr << "Cannot open the engine file: " + engine_path << std::endl;
		return false;
	}

	file.seekg(0, std::ios::end);
	size_t size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> data(size);
	file.read(data.data(), size);
	file.close();

	runtime_ = nvinfer1::createInferRuntime(logger_);
	if (!runtime_)
	{
		std::cerr << "ERROR: Create runtime failed" << std::endl;
		return false;
	}

	engine_ = runtime_->deserializeCudaEngine(data.data(), size);
	if (!engine_)
	{
		std::cerr << "ERROR: Deserialized engine failed" << std::endl;
		return false;
	}

	isLoaded_ = true;
	std::cout << "Engine successfully loaded" << std::endl;
	return true;
}

bool TRTInferEngine::createContext()
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

	return true;
}

bool TRTInferEngine::allocateBuffer(const std::vector<size_t>& bufferSizes)
{
	if (!context_)
	{
		std::cerr << "ERROR: Context is not ready" << std::endl;
		return false;
	}

	buffers_.clear();
	bufferSizes_ = bufferSizes;

	for (size_t i = 0; i < bufferSizes.size(); i++)
	{
		void* buffer = nullptr;
		cudaError err = cudaMalloc(&buffer, bufferSizes[i]);
		if (err != cudaSuccess)
		{
			std::cerr << "ERROR: Failed to allocate CUDA memory for buffer " << std::endl;
			return false;
		}
		buffers_.push_back(buffer);
	}

	return true;
}

bool TRTInferEngine::inference()
{
	if (!context_ || buffers_.empty())
	{
		std::cerr << "ERROR: Context or buffer are not ready" << std::endl;
		return false;
	}

	return context_->executeV2(buffers_.data());
}

void TRTInferEngine::cleanUp()
{
	for (auto& buffer : buffers_)
	{
		if (buffer)
		{
			cudaFree(buffer);
		}
	}
	buffers_.clear();

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
}

// =================== 图像前后处理类实现 ==================
ImageProcessor::ImageProcessor(int width, int height)
	:width_(width)
	, height_(height)
	, imagesLoaded_(false)
{

}

bool ImageProcessor::loadImages(const std::string& img_path, const std::string& lmk_path, const std::string& mask_path)
{
	originalImage_ = cv::imread(img_path);
	landmarkImage_ = cv::imread(lmk_path, cv::IMREAD_GRAYSCALE);
	maskImage_ = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);

	if (originalImage_.empty() || landmarkImage_.empty() || maskImage_.empty())
	{
		std::cerr << "ERROR: Load input failed" << std::endl;
		return false;
	}

	cv::resize(originalImage_, originalImage_, cv::Size(width_, height_));
	cv::resize(landmarkImage_, landmarkImage_, cv::Size(width_, height_));
	cv::resize(maskImage_, maskImage_, cv::Size(width_, height_));

	cv::cvtColor(originalImage_, originalImage_, cv::COLOR_BGR2RGB);

	cv::threshold(maskImage_, maskImage_, 0, 255, cv::THRESH_BINARY);

	imagesLoaded_ = true;
	std::cout << "Images loaded and preprocessed successfully" << std::endl;
	return true;
}

void ImageProcessor::normalizeImages()
{
	cv::Mat img_float, lmk_float, mask_float;

	originalImage_.convertTo(img_float, CV_32F, 1.0 / 127.5, -1.0);
	landmarkImage_.convertTo(lmk_float, CV_32F, 1.0 / 255.0);
	maskImage_.convertTo(mask_float, CV_32F, 1.0 / 255.0);

	imageData_.assign((float*)img_float.datastart, (float*)img_float.dataend);
	landmarkData_.assign((float*)lmk_float.datastart, (float*)lmk_float.dataend);
	maskData_.assign((float*)mask_float.datastart, (float*)mask_float.dataend);
}

void ImageProcessor::convertHWCtoCHW()
{
	std::vector<float> img_chw(3 * width_ * height_);
	for (int h = 0; h < height_; h++)
	{
		for (int w = 0; w < width_; w++)
		{
			int idx = h * width_ + w;
			for (int c = 0; c < 3; c++)
			{
				img_chw[c * width_ * height_ + idx] = imageData_[idx * 3 + c];
			}
		}
	}

	imageData_ = std::move(img_chw);
}

InputData ImageProcessor::preprocess()
{
	if (!imagesLoaded_)
	{
		std::cerr << "ERROR: Image not Loaded" << std::endl;
		return InputData{};
	}

	normalizeImages();
	convertHWCtoCHW();

	InputData input_data;
	input_data.image = std::move(imageData_);
	input_data.imageSize = 3 * height_ * width_ * sizeof(float);
	input_data.landmark = std::move(landmarkData_);
	input_data.landmarkSize = 1 * height_ * width_ * sizeof(float);
	input_data.mask = std::move(maskData_);
	input_data.maskSize = 1 * height_ * width_ * sizeof(float);

	return input_data;
}

OutputData ImageProcessor::postprocess(const std::vector<float>& modelOutput)
{
	OutputData output_data;
	output_data.result = modelOutput;

	cv::Mat output_float(height_, width_, CV_32FC3);

	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < height_; ++h) {
			for (int w = 0; w < width_; ++w) {
				float val = modelOutput[c * 256 * 256 + h * 256 + w] * 255.0f;
				if (val < 0) val = 0;
				if (val > 255) val = 255;
				output_float.at<cv::Vec3f>(h, w)[c] = val;
			}
		}
	}

	output_float.convertTo(output_float, CV_8UC3);
	cv::cvtColor(output_float, resultImage_, cv::COLOR_RGB2BGR);
	output_data.resultImage = resultImage_;
	return output_data;
}

void ImageProcessor::showResult(const std::string& WindowName)
{
	if (!resultImage_.empty())
	{
		cv::imshow(WindowName, resultImage_);
		cv::waitKey(0);
	}
	else
	{
		std::cerr << "Error: No result image to display" << std::endl;
	}
}

void ImageProcessor::saveResult(const std::string& path)
{
	if (!resultImage_.empty())
	{
		cv::imwrite(path, resultImage_);
		std::cout << "Result saved to: " << path << std::endl;
	}
	else {
		std::cerr << "Error: No result image to save" << std::endl;
	}
}

// ================== Inpaingting类实现 =======================
Inpainting::Inpainting()
	: w_(256)
	, h_(256)
	, isLoaded(false)
	, isInitialized(false)
{
	Engine_ = std::make_unique<TRTInferEngine>();
	Processor_ = std::make_unique<ImageProcessor>(w_, h_);
}

Inpainting::~Inpainting()
{
	Engine_->cleanUp();
}

bool Inpainting::Init(const std::string& path)
{
	if (!Engine_->loadEngine(path)) return false;
	if (!Engine_->createContext()) return false;

	size_t img_size = 3 * h_ * w_ * sizeof(float);
	size_t lmk_size = h_ * w_ * sizeof(float);
	size_t mask_size = h_ * w_ * sizeof(float);
	size_t out_size = 3 * h_ * w_ * sizeof(float);

	std::vector<size_t> bufferSizes{ img_size, lmk_size, mask_size, out_size };

	if (!Engine_->allocateBuffer(bufferSizes)) return false;

	isInitialized = true;
	std::cout << "FaceInpainting initialized successfully" << std::endl;
	return true;
}

bool Inpainting::loadImg(const std::string& img_path, const std::string& lmk_path, const std::string& mask_path)
{
	if (!Processor_->loadImages(img_path, lmk_path, mask_path)) return false;

	isLoaded = true;
	return true;
}

bool Inpainting::copyToDevice()
{
	if (!isLoaded || !isInitialized)
	{
		std::cerr << "ERROR: Not initialized or images not loaded" << std::endl;
		return false;
	}

	input_ = Processor_->preprocess();

	cudaError_t err;
	err = cudaMemcpy(Engine_->getBuffer(0), input_.image.data(), input_.imageSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) return false;

	err = cudaMemcpy(Engine_->getBuffer(1), input_.landmark.data(), input_.landmarkSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) return false;

	err = cudaMemcpy(Engine_->getBuffer(2), input_.mask.data(), input_.maskSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) return false;

	return true;
}

bool Inpainting::copyFromDevice()
{
	size_t outSize = 3 * h_ * w_ * sizeof(float);
	std::vector<float> outData(outSize / sizeof(float));

	cudaError err = cudaMemcpy(outData.data(), Engine_->getBuffer(3), outSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) return false;

	output_ = Processor_->postprocess(outData);
	return true;
}

bool Inpainting::runInference()
{
	if (!isInitialized || !isLoaded)
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
	if (!Engine_->inference())
	{
		std::cerr << "ERROR: Failed to infer" << std::endl;
		return false;
	}
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

	std::cout << "\n==================== 耗时统计 ====================\n";
	std::cout << "预处理 + 数据拷贝: " << pre_time << " ms\n";
	std::cout << "模型推理耗时:     " << infer_time << " ms\n";
	std::cout << "数据回传 + 后处理: " << post_time << " ms\n";
	std::cout << "==================================================\n";
	std::cout << "总 pipeline 耗时:  " << total_time << " ms\n";
	std::cout << "==================================================\n\n";

	std::cout << "Inference completed successfully" << std::endl;
	return true;
}

bool Inpainting::saveResult(const std::string& path)
{
	Processor_->saveResult(path);
	return true;
}

void Inpainting::showResult()
{
	Processor_->showResult("FaceInpainting result");
}