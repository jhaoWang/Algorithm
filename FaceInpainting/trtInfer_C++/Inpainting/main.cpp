#include "InpaintingTrT_module.h"
#include <iostream>

int main()
{
	Inpainting inp;

	std::string path = "E:/code_git/Algorithm/FaceInpainting/trtInfer_python/inpaint.engine";

	if (!inp.Init(path)) {
		std::cerr << "ERROR: Failed to init inpainting app" << std::endl;
		return -1;
	}

	std::string imagePath = "E:/pythonCode/lafin-master/lafin-master/examples/images/195579.jpg";
	std::string landmarkPath = "E:/pythonCode/lafin-master/lafin-master/checkpoints/results/landmark_inpaint/landmark/195579.png";
	std::string maskPath = "E:/pythonCode/lafin-master/lafin-master/examples/masks/11257.png";

	if (!inp.loadImg(imagePath, landmarkPath, maskPath))
	{
		std::cerr << "ERROR: Failed to load images" << std::endl;
		return -1;
	}

	if (!inp.runInference())
	{
		std::cerr << "Inference failed" << std::endl;
		return -1;
	}

	inp.showResult();
	std::cout << "Inpainting successfully!" << std::endl;
	return 0;
}