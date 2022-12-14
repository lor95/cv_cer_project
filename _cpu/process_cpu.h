#pragma once
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat main_logic_cpu(Mat mainframe, cv::CascadeClassifier target_cascade, double f, double w);