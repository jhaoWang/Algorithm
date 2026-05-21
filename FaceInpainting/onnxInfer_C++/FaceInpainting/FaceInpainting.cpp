#include "Inpainting_module.h"
#include <iostream>;

int main()
{
	Inpainting inp;

	std::string model_path = "E:\\pythonCode\\lafin-master\\lafin-master\\inpaint.onnx";
	if (!inp.Init(model_path))
	{
		std::cerr << "Failed to initialize application!" << std::endl;
		return -1;
	}

	inp.setInputPath("E:\\pythonCode\\lafin-master\\lafin-master\\examples\\images\\195579.jpg",
		"E:\\pythonCode\\lafin-master\\lafin-master\\checkpoints\\results\\landmark_inpaint\\landmark\\195579.png",
		"E:\\pythonCode\\lafin-master\\lafin-master\\examples\\masks\\11257.png");

	if (inp.run())
	{
		std::cout << "Inpainting completed successfully!" << std::endl;
	}
	else
	{
		std::cerr << "Inpainting failed!" << std::endl;
		return -1;
	}
	return 0;
}