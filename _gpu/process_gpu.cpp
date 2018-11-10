#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <iomanip>

#include "process_gpu.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

static std::vector<Point> pos;
static int fail_counter = 0;

GpuMat process_gpu(Mat mainframe, cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu);
GpuMat main_logic_gpu(Mat frame, cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu);

GpuMat process_gpu(Mat mainframe, cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu) {
	putText(mainframe, "USING: GPU", Point2f(10, 20), FONT_HERSHEY_DUPLEX, 0.9, Scalar(0, 0, 255, 255));
	//cv::cuda::printShortCudaDeviceInfo(cv::cuda::getCudaEnabledDeviceCount());
	return main_logic_gpu(mainframe, target_cascade_gpu);
}

GpuMat main_logic_gpu(Mat frame, cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu) {
	if (pos.size() >= 30) { // max number of points evaluated in trajectory
		pos.erase(pos.begin()); // delete first point in trajectory
	}
	std::vector<Rect> targets; // target matrixes
	std::vector<Point> centers; // target centers
	std::vector<Point> dist_arr;
	std::vector<Point> distj_arr;
	GpuMat frame_gpu(frame);
	GpuMat fgray;
	GpuMat objbuf;
	cv::cuda::cvtColor(frame_gpu, fgray, COLOR_BGR2GRAY);
	cv::cuda::equalizeHist(fgray, fgray);
	cascade_gpu->setScaleFactor(1.1);
	cascade_gpu->setMinNeighbors(2);
	cascade_gpu->setMinObjectSize(cv::Size(30, 30));
	cascade_gpu->detectMultiScale(fgray, objbuf); // detect targets and store them in objects buffer (rectangles)  NOTE: Cuda detectMultiScale has only one form, without the confidence levels return
	cascade_gpu->convert(objbuf, targets);  // convert objects array from internal representation to standard vector 'targets'
	size_t i = targets.size();
	if (i == 0) {
		fail_counter++; // increment
		putText(frame, "No target is tracked.", Point2f(10, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
	}
	for (size_t j = 0; j < i; j++) { // draw targets on frame
		fail_counter = 0; // reset
		Point center(targets[j].x + targets[j].width / 2, targets[j].y + targets[j].height / 2);
		centers.push_back(center);
		rectangle(frame, Size(targets[j].x + targets[j].width, targets[j].y + targets[j].height),
			Point(targets[j].x, targets[j].y), Scalar(0, 255, 0), 1, 8, 0); // draw target
		if (i == 1) { // only 1 target is tracked
			pos.push_back(centers[j]);
		}
	}
	if (i > 1 && !pos.empty()) { // if more than a target is tracked insert the more probable center in trajectory
		Point last_tr = Point(pos[pos.size() - 1].x, pos[pos.size() - 1].y);
		dist_arr.push_back(last_tr);
		dist_arr.push_back(centers[0]);
		double dist = cv::cuda::norm(dist_arr, NORM_L2); // distance between latest tr and centers tracked
		size_t index = 0;
		for (size_t j = 1; j < i; j++) {
			distj_arr.push_back(last_tr);
			distj_arr.push_back(centers[j]);
			if (cv::cuda::norm(distj_arr, NORM_L2) < dist) {
				dist = cv::cuda::norm(distj_arr, NORM_L2);
				index = j;
			}
		}
		pos.push_back(centers[index]);
	}
	if (fail_counter <= 15) {
		for (size_t j = 1; j < pos.size(); j++) {
			line(frame, pos[j], pos[j - 1], Scalar(0, 255, 150), 2, 8, 0); // draw trajectory
		}
	}
	else {
		pos.clear();
	}
	// write frame data
	ostringstream target_position;
	Scalar info_color;
	if (!pos.empty() && i != 0) {
		info_color = Scalar(0, 255, 255);
		target_position << "Position:{x = " << pos.back().x << "; y = " << pos.back().y << "; z = ???}";
	}
	else {
		info_color = Scalar(0, 0, 255);
		target_position << "No target is tracked.";
	}
	putText(frame, target_position.str(),
		Point2f(10, 65), FONT_HERSHEY_DUPLEX, 0.45, info_color); // position info
	frame_gpu.upload(frame);
	return frame_gpu;
}