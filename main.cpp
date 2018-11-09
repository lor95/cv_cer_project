// this is the master file. It initializes the project 
#include <iostream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

#include "_cpu\process_cpu.h"
#include "_gpu\process_gpu.h"
#include "_calib\calibration.h"

#define CONTINUE 110 // n key
#define CALIBRATE 99 // c key
#define SWITCH 32 // space
#define EXIT 27 // esc

using namespace cv;
using namespace std;
namespace fs = std::tr2::sys;

bool calib_out_bool = false;
CascadeClassifier target_cascade;

int main( int argc, const char** argv ) {	

	cout << "CV_CER_PROJECT.\n\n";
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
	else {
		cout << "cascade.xml has been found.\n";
	}

	string data_path = "_data";
	ostringstream elem;
	for (auto & p : fs::directory_iterator(data_path)) { // look inside "_data" directory
		elem << p;
		if (elem.str().find("calibration_output.xml") != string::npos) {
			calib_out_bool = true; // if calibration_output.xml is found
		}
		elem.str("");
	}
	if (calib_out_bool) {
		cout << "calibration_output.xml has been found.\n\n";
	}
	else {
		cout << "calibration_output.xml has not been found.\n\n";
	}
	
	Mat mainframe; // main frame from camera
	VideoCapture capture(0); // open the first camera
	if (!capture.isOpened()) {
		cerr << "ERROR: Can't initialize camera capture." << endl;
		return 2;
	}

	cout << "Opening camera...\n\n";

	if (calib_out_bool) {
		cout << "CALIBRATION OPTIONS : Press 'n' to continue.Press 'c' to RECALIBRATE.\nPress ESC to exit.\n\n";
	}
	else {
		cout << "CALIBRATION OPTIONS : Press 'n' to continue.Press 'c' to CALIBRATE.\nPress ESC to exit.\n\n";
	}
	
	while (true) { // first loop for calibration options
		int k = waitKey(1); //waits for a key indefinitely
		if (k == CONTINUE) {
			break;
		}
		if (k == CALIBRATE) {
			// calibrate()
			cout << "doing some funny calibrations\n\n";
		}
		if (k == EXIT) {
			return 0; // exit
		}

		capture >> mainframe;
		if (mainframe.empty()) {
			cerr << "ERROR: Can't grab camera frame." << endl;
			return 3;
		}
		mainframe = _calibration_wait(mainframe);
		imshow("window_name", mainframe);
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
			mainframe = process_cpu( mainframe, target_cascade );
			imshow("window_name", mainframe); // display frame
		}
		else { // code for GPU
			sw = !sw;
			mainframe = process_gpu( mainframe, target_cascade );
			imshow("window_name", mainframe); // display frame (not processed yet)
		}
	}
	return 0;
}