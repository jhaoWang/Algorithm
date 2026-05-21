#include "LmkDetect_module.h"
#include <iostream>

int main()
{
	Detection detector;

	std::string model_path = "landmark.onnx";
	if (!detector.Init(model_path))
	{
		std::cerr << "Failed to init detector" << std::endl;
		return -1;
	}

	detector.setInputPath("E:\\pythonCode\\lafin-master\\lafin-master\\examples\\images\\195579.jpg",
		"E:\\pythonCode\\lafin-master\\lafin-master\\examples\\masks\\11257.png");

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