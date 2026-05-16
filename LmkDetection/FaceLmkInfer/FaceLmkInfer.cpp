#include <iostream>
#include <fstream>
#include <vector>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "opencv2\opencv.hpp"

const int INPUT_SIZE = 256;
const int LANDMARK_NUM = 68;

const std::string ENGINE_PATH = "landmark.plan";
const std::string IMG_PATH = "E:\\pythonCode\\lafin-master\\lafin-master\\examples\\images\\195579.jpg";
const std::string MASK_PATH = "E:\\pythonCode\\lafin-master\\lafin-master\\examples\\masks\\08012.png";
const std::string SAVE_RESULT = "result_landmark_trt.png";


#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

class Logger :public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity == Severity::kWARNING) {
			std::cout << "Warning" << msg << std::endl;
		}
		else if (severity == Severity::kERROR)
		{
			std::cout << "Error" << msg << std::endl;
		}
	}
}gLogger;

std::vector<char> readEnginFile(const std::string& path)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open the file:" + path);
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size))
	{
		throw std::runtime_error("Filed to read engin file");
	}

	std::cout << "Load engin file, size" << size << "bytes" << std::endl;
	return buffer;
}

struct HostDeviceMem {
	float* host;
	void* device;
	size_t size;
};

class LmkInfer {
public:
	LmkInfer(const std::string& path)
	{
		std::vector<char> engineData = readEnginFile(path);

		runtime = nvinfer1::createInferRuntime(gLogger);
		if (!runtime)
		{
			throw std::runtime_error("Failed to create runtime");
		}

		engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
		if (!engine)
		{
			throw std::runtime_error("Failed to deserialize enging");
		}

		context = engine->createExecutionContext();
		if (!context)
		{
			throw std::runtime_error("Failed to create context");
		}

		allocateBuffers();

		CHECK_CUDA(cudaStreamCreate(&stream));

		std::cout << "TensorRT inference engine initialized successfully!" << std::endl;
		std::cout << "Input tensors: " << inputTensorNames.size() << std::endl;
		std::cout << "Output tensors: " << outputTensorNames.size() << std::endl;
	}

	~LmkInfer() {
		if (stream) {
			cudaStreamDestroy(stream);
		}

		for (auto& mem : inputs) {
			if (mem.host) {
				cudaFreeHost(mem.host);
			}

			if (mem.device) {
				cudaFree(mem.device);
			}
		}

		for (auto& mem : outputs) {
			if (mem.host) {
				cudaFreeHost(mem.host);
			}

			if (mem.device) {
				cudaFree(mem.device);
			}
		}

		if (context) delete context;
		if (engine) delete engine;
		if (runtime) delete runtime;
	}

	void allocateBuffers() {
		int numIOTensors = engine->getNbIOTensors();
		for (int i = 0; i < numIOTensors; i++)
		{
			const char* tensorName = engine->getIOTensorName(i);
			nvinfer1::TensorIOMode mode = engine->getTensorIOMode(tensorName);
			nvinfer1::Dims dims = context->getTensorShape(tensorName);

			size_t size = 1;
			for (int j = 0; j < dims.nbDims; j++) {
				size *= dims.d[j];
			}

			nvinfer1::DataType dtype = engine->getTensorDataType(tensorName);
			size_t typeSize = (dtype == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(float);
			size_t bufferSize = size * typeSize;

			void* hostMem = nullptr;
			CHECK_CUDA(cudaMallocHost(&hostMem, bufferSize));

			void* deviceMem = nullptr;
			CHECK_CUDA(cudaMalloc(&deviceMem, bufferSize));

			HostDeviceMem mem;
			mem.host = static_cast<float*>(hostMem);
			mem.device = deviceMem;
			mem.size = size;

			if (mode == nvinfer1::TensorIOMode::kINPUT) {
				inputs.push_back(mem);
				inputTensorNames.push_back(tensorName);
			}
			else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
				outputs.push_back(mem);
				outputTensorNames.push_back(tensorName);
			}
		}

		if (inputs.size() != 2) {
			std::cerr << "Warning: Expected 2 inputs, got " << inputs.size() << std::endl;
		}

	}

	void preprocess(const cv::Mat& srcImg, const cv::Mat& srcMask, std::vector<float>& inputFeed, std::vector<float>& inputMask)
	{
		cv::Mat img;
		cv::resize(srcImg, img, cv::Size(INPUT_SIZE, INPUT_SIZE));

		// 归一化到0-1，并转化为float类型
		cv::Mat imgFloat;
		img.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

		// 转换为CHW格式
		std::vector<cv::Mat> channels(3);
		cv::split(imgFloat, channels);
		inputFeed.resize(3 * INPUT_SIZE * INPUT_SIZE);
		for (int c = 0; c < 3; c++)
		{
			std::memcpy(inputFeed.data() + c * INPUT_SIZE * INPUT_SIZE, channels[c].data, INPUT_SIZE * INPUT_SIZE * sizeof(float));
		}

		cv::Mat mask;
		cv::resize(srcMask, mask, cv::Size(INPUT_SIZE, INPUT_SIZE));

		cv::Mat maskFloat;
		mask.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);

		inputMask.resize(INPUT_SIZE * INPUT_SIZE);
		std::memcpy(inputMask.data(), maskFloat.data, INPUT_SIZE * INPUT_SIZE * sizeof(float));

		// 对图像添加掩码
		for (int i = 0; i < 3 * INPUT_SIZE * INPUT_SIZE; i++) {
			int indexpix = i % (INPUT_SIZE * INPUT_SIZE);
			float maskVal = inputMask[indexpix];
			inputFeed[i] = inputFeed[i] * (1.0f - maskVal) + maskVal;
		}
	}

	std::vector<float> infer(const std::vector<float>& inputFeed, const std::vector<float>& inputMask)
	{
		if (inputs.size() < 2)
		{
			throw std::runtime_error("Invalid number of inputs");
		}

		size_t expectedFeedSize = 3 * INPUT_SIZE * INPUT_SIZE;
		size_t expectedMaskSize = INPUT_SIZE * INPUT_SIZE;
		if (inputFeed.size() != expectedFeedSize) {
			throw std::runtime_error("Input feed size mismatch");
		}
		if (inputMask.size() != expectedMaskSize) {
			throw std::runtime_error("Input mask size mismatch");
		}

		// 推理的核心过程
		// 1.复制数据到主机内存
		std::memcpy(inputs[0].host, inputFeed.data(), inputFeed.size() * sizeof(float));
		std::memcpy(inputs[1].host, inputMask.data(), inputMask.size() * sizeof(float));

		// 2.异步拷贝到内存
		for (size_t i = 0; i < inputs.size(); i++) {
			CHECK_CUDA(cudaMemcpyAsync(inputs[i].device, inputs[i].host,
				inputs[i].size * sizeof(float),
				cudaMemcpyHostToDevice, stream));
		}

		// 3.设置张量地址
		for (size_t i = 0; i < inputTensorNames.size(); i++) {
			context->setTensorAddress(inputTensorNames[i].c_str(), inputs[i].device);
		}
		for (size_t i = 0; i < outputTensorNames.size(); i++) {
			context->setTensorAddress(outputTensorNames[i].c_str(), outputs[i].device);
		}

		// 4.执行推理
		bool success = context->enqueueV3(stream);
		if (!success) {
			throw std::runtime_error("Failed to enqueue inference");
		}

		// 5.将结果拷贝回主机（异步）
		for (size_t i = 0; i < outputs.size(); i++) {
			CHECK_CUDA(cudaMemcpyAsync(outputs[i].host, outputs[i].device,
				outputs[i].size * sizeof(float),
				cudaMemcpyDeviceToHost, stream));
		}

		// 6.同步流
		CHECK_CUDA(cudaStreamSynchronize(stream));

		std::vector<float> result(outputs[0].host, outputs[0].host + outputs[0].size);
		return result;
	}

	size_t getOutputSize() const {
		return outputs.empty() ? 0 : outputs[0].size;
	}
private:
	Logger gLogger;
	nvinfer1::IRuntime* runtime = nullptr;
	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;

	std::vector<HostDeviceMem> inputs;
	std::vector<HostDeviceMem> outputs;
	cudaStream_t stream;

	std::vector<std::string> inputTensorNames;
	std::vector<std::string> outputTensorNames;
};

void drawLandmarks(const cv::Mat& originalImg, const std::vector<float>& landmarks,
	int landmarkNum, int inputSize, const std::string& savePath) {
	int h = originalImg.rows;
	int w = originalImg.cols;

	// 重塑为 [landmark_num, 2]
	std::vector<cv::Point2f> points(landmarkNum);
	for (int i = 0; i < landmarkNum; i++) {
		float x = landmarks[i * 2];
		float y = landmarks[i * 2 + 1];

		// 边界裁剪
		x = std::min(std::max(x, 0.0f), static_cast<float>(inputSize - 1));
		y = std::min(std::max(y, 0.0f), static_cast<float>(inputSize - 1));

		// 缩放到原始图像尺寸
		float scaleX = static_cast<float>(w) / inputSize;
		float scaleY = static_cast<float>(h) / inputSize;

		points[i].x = x * scaleX;
		points[i].y = y * scaleY;
	}

	// 创建黑色背景
	cv::Mat result = cv::Mat::zeros(h, w, CV_8UC1);

	// 绘制关键点
	for (const auto& pt : points) {
		cv::circle(result, pt, 2, cv::Scalar(255), -1);
	}

	// 保存结果
	cv::imwrite(savePath, result);
	std::cout << "Result saved to " << savePath << std::endl;

	// 显示结果
	cv::imshow("Landmark Result TRT", result);
	cv::waitKey(0);
}

int main(int argc, char** argv) {
	try {
		std::cout << "Loading TensorRT engine from: " << ENGINE_PATH << std::endl;
		LmkInfer inference(ENGINE_PATH);

		std::cout << "Loading image: " << IMG_PATH << std::endl;
		cv::Mat srcImg = cv::imread(IMG_PATH);
		if (srcImg.empty()) {
			throw std::runtime_error("Failed to load image: " + IMG_PATH);
		}

		std::cout << "Loading mask: " << MASK_PATH << std::endl;
		cv::Mat srcMask = cv::imread(MASK_PATH, cv::IMREAD_GRAYSCALE);
		if (srcMask.empty()) {
			throw std::runtime_error("Failed to load mask: " + MASK_PATH);
		}

		std::cout << "Original image size: " << srcImg.cols << "x" << srcImg.rows << std::endl;
		std::cout << "Mask size: " << srcMask.cols << "x" << srcMask.rows << std::endl;

		std::cout << "Preprocessing..." << std::endl;
		std::vector<float> inputFeed;
		std::vector<float> inputMask;
		inference.preprocess(srcImg, srcMask, inputFeed, inputMask);

		std::cout << "Input feed size: " << inputFeed.size() << std::endl;
		std::cout << "Input mask size: " << inputMask.size() << std::endl;

		std::cout << "Running inference..." << std::endl;

		for (int i = 0; i < 5; i++) {
			inference.infer(inputFeed, inputMask);
		}

		auto start = std::chrono::high_resolution_clock::now();
		std::vector<float> landmarks = inference.infer(inputFeed, inputMask);
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;

		std::cout << "Postprocessing..." << std::endl;
		std::cout << "Landmarks size: " << landmarks.size() << std::endl;

		if (landmarks.size() != static_cast<size_t>(LANDMARK_NUM * 2)) {
			std::cout << "Warning: Expected " << LANDMARK_NUM * 2
				<< " values, got " << landmarks.size() << std::endl;
		}

		drawLandmarks(srcImg, landmarks, LANDMARK_NUM, INPUT_SIZE, SAVE_RESULT);
		std::cout << "\nFirst 10 landmarks:" << std::endl;
		for (int i = 0; i < std::min(10, LANDMARK_NUM); i++) {
			std::cout << "  Point " << i << ": ("
				<< landmarks[i * 2] << ", " << landmarks[i * 2 + 1] << ")" << std::endl;
		}

		std::cout << "\nProgram completed successfully!" << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}