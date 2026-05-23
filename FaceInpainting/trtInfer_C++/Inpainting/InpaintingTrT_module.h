#pragma once
#include <vector>
#include <string>
#include <opencv2\opencv.hpp>
#include <NvInfer.h>
#include <memory>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct PreprocessParam {
	int w_target = 256;
	int h_target = 256;
	float imgScale_min = -1.0f;
	float imgScale_max = 1.0f;
	float maskScale_min = 0.0f;
	float maskScale_max = 1.0f;
};

class Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING)
		{
			std::cout << "[TRT]" << msg << std::endl;
		}
	}
};

class TRTInferEngine
{
public:
	TRTInferEngine();
	~TRTInferEngine();
	bool loadEngine(const std::string& engine_path);
	bool createContext();
	bool allocateBuffer(const std::vector<size_t>& bufferSizes);
	void* getBuffer(int index) const { return buffers_[index]; }
	nvinfer1::IExecutionContext* getContext() const { return context_; }
	bool inference();
	void cleanUp();
private:
	Logger logger_;
	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	nvinfer1::IRuntime* runtime_;

	std::vector<void*> buffers_;
	std::vector<size_t> bufferSizes_;

	bool isLoaded_;
};

struct InputData {
	std::vector<float> image;
	std::vector<float> landmark;
	std::vector<float> mask;
	size_t imageSize;
	size_t landmarkSize;
	size_t maskSize;
};

struct OutputData {
	std::vector<float> result;
	cv::Mat resultImage;
};

class ImageProcessor {
public:
    ImageProcessor(int width = 256, int height = 256);
    ~ImageProcessor() = default;

    bool loadImages(const std::string& imagePath, const std::string& landmarkPath, const std::string& maskPath);

    InputData preprocess();
    OutputData postprocess(const std::vector<float>& modelOutput);

    void saveResult(const std::string& outputPath);
    void showResult(const std::string& windowName = "Result");

    // Getters
    cv::Mat getOriginalImage() const { return originalImage_; }
    cv::Mat getResultImage() const { return resultImage_; }

private:
    int width_;
    int height_;

    cv::Mat originalImage_;
    cv::Mat landmarkImage_;
    cv::Mat maskImage_;
    cv::Mat resultImage_;

    void normalizeImages();
    void convertHWCtoCHW();
    /*void validateImages();*/

    std::vector<float> imageData_;
    std::vector<float> landmarkData_;
    std::vector<float> maskData_;

    bool imagesLoaded_;
};

class Inpainting {
public:
	Inpainting();
	~Inpainting();
	bool Init(const std::string& modelpath);
	bool loadImg(const std::string& img_path, const std::string& lmk_path, const std::string& mask_path);
	bool runInference();
	bool saveResult(const std::string& outputPath);
	void showResult();


private:
	std::unique_ptr<TRTInferEngine> Engine_;
	std::unique_ptr<ImageProcessor> Processor_;

	InputData input_;
	OutputData output_;

	int w_;
	int h_;

	bool isInitialized;
	bool isLoaded;

	bool copyToDevice();
	bool copyFromDevice();
};

