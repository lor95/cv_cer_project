#include <iostream>
#include <time.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include "process_gpu.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

static std::vector<Point3d> pos;
static int fail_counter = 0;

GpuMat main_logic_gpu(Mat frame, cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu, double focal_length, double r_width);

GpuMat main_logic_gpu(Mat frame, cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu, double focal_length, double r_width) {

	clock_t t0 = clock();

	putText(frame, "USING: GPU", Point2f(10, 20), FONT_HERSHEY_DUPLEX, 0.9, Scalar(0, 0, 255, 255));

	if (pos.size() >= 30) { // max number of points evaluated in trajectory
		pos.erase(pos.begin()); // delete first point in trajectory
	}
	std::vector<Rect> targets; // target matrixes
	std::vector<Point> centers; // target centers
	std::vector<double> distances; // target distances
	size_t target = 0; // index of the target
	GpuMat frame_gpu(frame); // transferring data from system memory to gpu memory
	GpuMat fgray;
	cv::cuda::cvtColor(frame_gpu, fgray, COLOR_BGR2GRAY);
	cv::cuda::equalizeHist(fgray, fgray);
	cascade_gpu->setScaleFactor(1.1);
	cascade_gpu->setMinNeighbors(5);
	cascade_gpu->setMinObjectSize(cv::Size(30, 30));
	cascade_gpu->detectMultiScale(fgray, fgray); // detect targets and store them in objects buffer (rectangles)  NOTE: Cuda detectMultiScale has only one form, without the confidence levels return
	cascade_gpu->convert(fgray, targets);  // convert objects array from internal representation to standard vector 'targets'
	size_t i = targets.size();
	if (i == 0) { // if no target has been found
		fail_counter++;
		putText(frame, "No target is tracked.", Point2f(10, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
	}
	for (size_t j = 0; j < i; j++) {
		fail_counter = 0; // reset fail counter (targets are found)
		Point center(targets[j].x + targets[j].width / 2, targets[j].y + targets[j].height / 2);
		distances.push_back((focal_length * r_width) / (targets[j].width)); // calculate distance for each target
		centers.push_back(center);
		rectangle(frame, Size(targets[j].x + targets[j].width, targets[j].y + targets[j].height),
			Point(targets[j].x, targets[j].y), Scalar(0, 255, 0), 2, 8, 0); // draw targets on frame
		if (i == 1) { // if only 1 target is tracked
			pos.push_back(Point3d(centers[j].x, centers[j].y, distances[j]));
			target = j; // (j == i == 1)
		}
	}
	if (i > 1 && !pos.empty()) { // if more than a single target is tracked, insert the most likely position in trajectory
		Point3d last_tr = Point3d(pos[pos.size() - 1].x, pos[pos.size() - 1].y, pos[pos.size() - 1].z); // last position in trajectory
		double dist = norm(last_tr - Point3d(centers[0].x, centers[0].y, distances[0])); // distance between last position and centers tracked
		size_t index = 0; // index of the target
		for (size_t j = 1; j < i; j++) {
			if (norm(last_tr - Point3d(centers[j].x, centers[j].y, distances[j])) <= dist) {
				dist = norm(last_tr - Point3d(centers[j].x, centers[j].y, distances[j]));
				index = j;
			}
		}
		pos.push_back(Point3d(centers[index].x, centers[index].y, distances[index]));
	}
	if (fail_counter <= 15) { // draw trajectory
		for (size_t j = 1; j < pos.size(); j++) {
			int thickness;
			if (pos[j].z < 300) { // if 0 < distance < 29cm
				thickness = 3;
			}
			else if (pos[j].z < 550) { // if 30cm < distance < 54cm
				thickness = 2;
			}
			else thickness = 1; // if distance > 55cm
			line(frame, Point(pos[j].x, pos[j].y), Point(pos[j - 1].x, pos[j - 1].y), Scalar(0, 255, 150), thickness, 8, 0); // draw trajectory
		}
	}
	else { // if in more than 15 frames no object is found, clear trajectory
		pos.clear();
	}
	/** write frame data **/
	ostringstream dst; // distance to be shown
	ostringstream target_position; // position to be shown
	double _z; // z axis

	if (!targets.empty()) {
		double distance = pos[pos.size() - 1].z;
		dst << "Distance: " << (distance / 10) << "cm"; // convert from millimeters to centimeters
		_z = distance - 500; // z is 0 when distance = 500 mm
	}
	Scalar info_color;
	if (!pos.empty() && i != 0) {
		info_color = Scalar(0, 255, 255);
		target_position << "Position:{x = " << pos.back().x << "; y = " << pos.back().y << "; z = " << (int)_z << "}";
	}
	putText(frame, target_position.str(),
		Point2f(10, 65), FONT_HERSHEY_DUPLEX, 0.55, info_color); // position info
	putText(frame, dst.str(),
		Point2f(10, 90), FONT_HERSHEY_DUPLEX, 0.55, info_color); // distance info
	clock_t t1 = clock() - t0;
	ostringstream time;
	time << "Execution time: " << t1 << "ms";
	putText(frame, time.str(), Point2f(10, 115), FONT_HERSHEY_DUPLEX, 0.55, Scalar(0, 255, 255)); // write exec time
	frame_gpu.upload(frame);
	return frame_gpu;
}