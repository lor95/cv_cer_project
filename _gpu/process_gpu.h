#pragma once
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat process_gpu( Mat mainframe, CascadeClassifier target_cascade );