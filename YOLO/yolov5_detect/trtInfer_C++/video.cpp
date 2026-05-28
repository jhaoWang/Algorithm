#include "yolov5_det_TRT.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main()
{
	yoloDet det;
	std::string modelPath = "E:/code_git/Algorithm/YOLO/yolov5_detect/trtInfer_python/yolov5s.engine";

	if (!det.Init(modelPath)) {
		std::cerr << "Init failed!" << std::endl;
		return -1;
	}

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "open video failed" << std::endl;
		return -1;
	}

	FrameQueue qIn, qOut;
	yoloDet_Video videoInfer(det);
	videoInfer.start(qIn, qOut);

	std::thread captureThread([&]()
		{
			cv::Mat frame;
			while (cap.read(frame)) {
				qIn.push(frame);
			}
			qIn.stop();
		});

	cv::Mat showFrame;

	int frame_cnt = 0;
	double fps = 0.0;
	auto start_time = std::chrono::high_resolution_clock::now();
	const int calc_interval = 30;

	while (true) {
		if (qOut.pop(showFrame)) {
			frame_cnt++;
			if (frame_cnt % calc_interval == 0)
			{
				auto now = std::chrono::high_resolution_clock::now();
				double duration = std::chrono::duration<double>(now - start_time).count();
				fps = frame_cnt / duration;
				frame_cnt = 0;
				start_time = now;
			}

			cv::putText(showFrame, "FPS: " + std::to_string(fps),
				cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
				cv::Scalar(0, 0, 255), 2);
			imshow("High Throughput YOLO", showFrame);
		}

		if (cv::waitKey(1) == 27) {
			qIn.stop();
			qOut.stop();
			break;
		}
	}

	captureThread.join();
	videoInfer.stop();
	cap.release();
	cv::destroyAllWindows();

	return 0;
}