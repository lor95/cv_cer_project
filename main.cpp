// this is the master file. It initializes the project 
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <filesystem>

#include "_cpu\process_cpu.h"
#include "_gpu\process_gpu.h"
#include "_calib\calibration.h"

#define SWITCH 32 // space
#define EXIT 27 // esc

#define W 190 //mm for face

using namespace cv;
using namespace cv::cuda;
using namespace std;
namespace fs = std::tr2::sys;

bool calib_out_bool = false;
double focal_length;
static cv::CascadeClassifier target_cascade;
static cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu;

int main( int argc, const char** argv ) {	
	cout << "CV_CER_PROJECT.\n\n";

	string data_path = "_data";
	ostringstream elem;
	for (auto p = fs::directory_iterator(data_path); p != fs::directory_iterator(); ++p) { // look inside "_data" directory
	//for (auto &p : fs::directory_iterator(data_path)) { // look inside "_data" directory
		elem << &p;
		//elem << p;
		if (elem.str().find("calibration_output.xml") != string::npos) {
			calib_out_bool = true; // if calibration_output.xml is found
		}
		elem.str("");
	}

	if (calib_out_bool) {
		cout << "calibration_output.xml has been found.\n\nOpening camera...\n\n";
		cout << "CALIBRATION OPTIONS : Press 'c' to RECALIBRATE, 'n' to continue.\n\n";
	}
	else {
		cout << "calibration_output.xml has not been found.\n\nOpening camera...\n\n";
		cout << "CALIBRATION OPTIONS : Press 'c' to CALIBRATE, 'n' to continue.\n\n";
	}

	focal_length = _calibration_init();

	cout << "focal length is: " << focal_length << endl;
	
	Mat mainframe; // main frame from camera
	GpuMat mainframe_gpu; // main frame from camera
	VideoCapture capture(0); // open the first camera
	if (!capture.isOpened()) {
		cerr << "ERROR: Can't initialize camera capture." << endl;
		return 2;
	}

	bool sw = false; // false = use CPU, true = use GPU (CUDA functions)

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

	cout << "Executing...\n\nCOMMANDS: Toggle SPACE button to switch between CPU and GPU mode.\nPress ESC to exit.\n\n";
	
	while (true) { // main loop
		int k = waitKey(1); //waits for a key indefinitely

		if (k == EXIT) {
			break; //if key = ESC exit
		}
		if (k == SWITCH) {
			sw = !sw;
		}

		// core
		capture >> mainframe; // read the next frame from camera
		if (mainframe.empty()) {
			cerr << "ERROR: Can't grab camera frame." << endl;
			return 3;
		}
		if (!sw) { // code for CPU
			namedWindow("cpu_window", WINDOW_AUTOSIZE);
			mainframe = process_cpu(mainframe, target_cascade, focal_length, W);
			imshow("cpu_window", mainframe); // display frame
		}
		else { // code for GPU
			namedWindow("gpu_window", WINDOW_OPENGL);
			resizeWindow("gpu_window", 640, 480);
			mainframe_gpu = process_gpu(mainframe, target_cascade_gpu, focal_length, W);
			imshow("gpu_window", mainframe_gpu); // display frame
		}
	}
	return 0;
}