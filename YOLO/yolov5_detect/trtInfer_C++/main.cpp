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

	/*std::string imagePath = "E:/code_git/Algorithm/YOLO/yolov5_detect/trtInfer_python/bus.jpg";*/
	cv::VideoCapture cap(0);

	cv::Mat frame;

	while (true)
	{
		// 1. 读一帧
		bool ret = cap.read(frame);
		if (!ret) {
			std::cout << "视频结束" << std::endl;
			break;
		}

		// 2. 直接把这一帧丢给你原来的推理
		// ---------------- 核心代码 ----------------
		det.setInput(frame);  // 直接赋值，不需要loadImg！

		if (!det.runInference()) {
			std::cerr << "推理失败" << std::endl;
			break;
		}

		// 3. 绘制结果
		det.DrawandshowResult();

		// 4. 按ESC退出
		if (cv::waitKey(1) == 27) break;
	}
	//if (!cap.isOpened()) {
	//	std::cerr << "ERROR: 无法打开摄像头/视频" << std::endl;
	//	return -1;
	//}

	//if (!det.loadImg(imagePath))
	//{
	//	std::cerr << "ERROR: Failed to load images" << std::endl;
	//	return -1;
	//}

	//if (!det.runInference())
	//{
	//	std::cerr << "Inference failed" << std::endl;
	//	return -1;
	//}
	//det.DrawandshowResult();
	//std::cout << "det successfully!" << std::endl;
	cap.release();
	cv::destroyAllWindows();
	return 0;
	return 0;
}