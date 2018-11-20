#pragma once
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace cv::cuda;

GpuMat main_logic_gpu(Mat mainframe, cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu, double focal_length, double r_width);