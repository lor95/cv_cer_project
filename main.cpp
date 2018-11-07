// this is the master file. It initializes the project 
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

#include "_cpu\process_cpu.h"
#include "_gpu\process_gpu.h"

using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;

CascadeClassifier target_cascade;

int main( int argc, const char** argv ) {

	fs::path p = "_data\\haarcascade_frontalface_default.xml"; // xml has to be in "<executable>/_data/"

	int ret = 0, loop = 1; // ret -> return, loop -> while
	bool sw = false; // false = use CPU, true = use GPU (CUDA functions)
	cout << "CV_cer_project.\n\nToggle SPACE button to switch between CPU and GPU mode.\n";
	cout << "Press ESC to exit.\n\nOpening camera...\n";

	CommandLineParser parser(argc, argv, // cpu
		"{help h||}"
		"{target_cascade|" + fs::system_complete(p).string() + "|}");
	String target_cascade_name = parser.get<string>("target_cascade");
	if (!target_cascade.load(target_cascade_name)) { // load the cascade
		cerr << "ERROR: Can't load target cascade." << endl;
		loop = 0;
		ret = 1;
	};
	
	Mat mainframe; // main frame from camera
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
			mainframe = process_cpu( mainframe, target_cascade );
			imshow("window_name", mainframe); // display frame
		}
		else { // code for GPU
			sw = !sw;
			mainframe = process_gpu( mainframe, target_cascade );
			imshow("window_name", mainframe); // display frame (not processed yet)
		}
	}
	return ret;
}