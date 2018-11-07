#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2\core\cuda.hpp>
#include <iostream>

#include "process_gpu.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

Mat process_gpu( Mat mainframe, CascadeClassifier target_cascade );

Mat process_gpu( Mat mainframe, CascadeClassifier target_cascade ) {
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getCudaEnabledDeviceCount());
	return mainframe;
}