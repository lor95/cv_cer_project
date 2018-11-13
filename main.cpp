// this is the master file. It initializes the project 
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <filesystem>

#include "_cpu\process_cpu.h"
#include "_gpu\process_gpu.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;
namespace fs = std::tr2::sys;

cv::CascadeClassifier target_cascade;
cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu;

int main(int argc, const char** argv) {

	int ret = 0, loop = 1; // ret -> return, loop -> while
	bool sw = false; // false = use CPU, true = use GPU (CUDA functions)
	cout << "CV_cer_project.\n\nToggle SPACE button to switch between CPU and GPU mode.\n";
	cout << "Press ESC to exit.\n\nOpening camera...\n";

	fs::path p = "_data\\haarcascade_frontalface_default.xml"; // xml has to be in "<executable>/_data/"
	CommandLineParser parser(argc, argv, // cpu
		"{help h||}"
		"{target_cascade|" + fs::system_complete(p).string() + "|}");
	String target_cascade_name = parser.get<string>("target_cascade");
	if (!target_cascade.load(target_cascade_name)) { // load the cascade
		cerr << "ERROR: Can't load target cascade." << endl;
		return 1;
	}

	fs::path n = "_data\\haarcascades_cuda\\haarcascade_frontalface_default.xml"; // xml has to be in "<executable>/_data/"
	CommandLineParser parser_gpu(argc, argv, // gpu
		"{help h||}"
		"{target_cascade_gpu|" + fs::system_complete(n).string() + "|}");
	String target_cascade_gpu_name = parser_gpu.get<string>("target_cascade_gpu");
	try
	{
		target_cascade_gpu = cv::cuda::CascadeClassifier::create(target_cascade_gpu_name); // load the cascade
	}
	catch (std::exception& e) //catches all types of exceptions, but will print an error if the cascade above fails to load
	{
		cerr << "ERROR: Can't load target cascade." << endl;
		return 1;
	}

	Mat mainframe; // main frame from camera
	GpuMat mainframe_gpu; // main frame from camera
	VideoCapture capture(0); // open the first camera
	if (!capture.isOpened()) {
		cerr << "ERROR: Can't initialize camera capture." << endl;
		loop = 0;
		ret = 1;
	}

	while (loop) {
		int k = waitKey(1); //waits for a key indefinitely
		if (k == 27) {
			break; //if key = ESC exit
		}
		if (k == 32) {
			sw = !sw;
		}

		// core
		capture >> mainframe; // read the next frame from camera
		if (mainframe.empty()) {
			cerr << "ERROR: Can't grab camera frame." << endl;
			break;
		}

		if (!sw) { // code for CPU
			namedWindow("cpu_window", WINDOW_AUTOSIZE);
			mainframe = process_cpu(mainframe, target_cascade);
			imshow("cpu_window", mainframe); // display frame
		}
		else { // code for GPU
			namedWindow("gpu_window", WINDOW_OPENGL);
			resizeWindow("gpu_window", 640, 480);
			mainframe_gpu = process_gpu(mainframe, target_cascade_gpu);
			imshow("gpu_window", mainframe_gpu); // display frame
		}
	}
	return ret;
}