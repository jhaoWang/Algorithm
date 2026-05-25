#include "yolov5Det_moudle.h"
#include <iostream>

int main()
{
	Detection detector;

	std::string model_path = "yolov5s.onnx";
	if (!detector.Init(model_path))
	{
		std::cerr << "Failed to init detector" << std::endl;
		return -1;
	}

	detector.setInputPath("bus.jpg");

	if (detector.run())
	{
		std::cout << "Detection finished successfully!" << std::endl;
	}
	else
	{
		std::cout << "Detection filed" << std::endl;
		return -1;
	}
	return 0;
}