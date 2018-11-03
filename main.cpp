//*this is the master file. It initializes the project 
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main() {
	int ret = 0, loop = 1; // ret -> return, loop -> while
	bool sw = false; // false = use CPU, true = use GPU (CUDA functions)
	cout << "CV_cer_project." << endl << "Opening camera..." << endl;
	
	Mat mainframe; // main frame from camera
	VideoCapture capture(0); // open the first camera
	if (!capture.isOpened())
	{
		cerr << "ERROR: Can't initialize camera capture" << endl;
		loop = 0;
		ret = 1;
	}

	while (true) {
		int k = waitKey(1); //waits for a key indefinitely
		if (k == 27) {
			break; //if key = ESC exit
		}
		if (k == 32) {
			sw = !sw;
		}

		// core
		capture >> mainframe; // read the next frame from camera
		if (mainframe.empty())
		{
			cerr << "ERROR: Can't grab camera frame." << endl;
			break;
		}

		if (!sw) { // code for CPU
			cout << "using CPU" << endl;
			imshow("Frame", mainframe); // display frame (not processed yet)
		}
		else { // code for GPU (doesn't use GPU yet)
			cout << "using GPU" << endl;
			imshow("Frame", mainframe); // display frame (not processed yet)
		}
	}
	return ret;
}