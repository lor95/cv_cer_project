#pragma once
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace std;

GpuMat process_gpu(Mat mainframe, cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu);