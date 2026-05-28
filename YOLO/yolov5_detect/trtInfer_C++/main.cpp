#include "yolov5_det_TRT.h"

#include <iostream>

int main()
{
	yoloDet det;
	std::string path = "E:/code_git/Algorithm/YOLO/yolov5_detect/trtInfer_python/yolov5s.engine";

	if (!det.Init(path)) {
		std::cerr << "ERROR: Failed to init det app" << std::endl;
		return -1;
	}

	std::string imagePath = "E:/code_git/Algorithm/YOLO/yolov5_detect/trtInfer_python/bus.jpg";

	if (!det.loadImg(imagePath))
	{
		std::cerr << "ERROR: Failed to load images" << std::endl;
		return -1;
	}

	if (!det.runInference())
	{
		std::cerr << "Inference failed" << std::endl;
		return -1;
	}
	det.DrawandshowResult();
	std::cout << "det successfully!" << std::endl;
	return 0;
}