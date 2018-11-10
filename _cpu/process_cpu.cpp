#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iostream>

#include "process_cpu.h"

using namespace std;
using namespace cv;

std::vector<Point> pos;
int fail_counter = 0;

Mat process_cpu( Mat mainframe, CascadeClassifier target_cascade, double focal_length, double r_width);
Mat main_logic_cpu( Mat frame, CascadeClassifier cascade, double focal_length, double r_width);

Mat process_cpu( Mat mainframe, CascadeClassifier target_cascade, double focal_length, double r_width) {
	putText(mainframe, "USING: CPU", Point2f(10, 20), FONT_HERSHEY_DUPLEX, 0.9, Scalar(0, 0, 255, 255));
	return main_logic_cpu ( mainframe, target_cascade, focal_length, r_width );
}

Mat main_logic_cpu ( Mat frame, CascadeClassifier cascade, double focal_length, double r_width) {

	if (pos.size() >= 30) { // max number of points evaluated in trajectory
		pos.erase(pos.begin()); // delete first point in trajectory
	}

	std::vector<Rect> targets; // target matrixes
	std::vector<Point> centers; // target centers
	size_t target = 0; // index of the target
	Mat fgray;
	cvtColor( frame, fgray, COLOR_BGR2GRAY );
	equalizeHist( fgray, fgray );
	cascade.detectMultiScale( fgray, targets, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) ); // detect targets
	size_t i = targets.size();
	if (i == 0) {
		fail_counter++; // increment
		putText(frame, "No target is tracked.", Point2f(10, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255));
	}
	for (size_t j = 0; j < i; j++ ) { // draw targets on frame
		fail_counter = 0; // reset
		Point center( targets[j].x + targets[j].width/2, targets[j].y + targets[j].height/2 );
		centers.push_back(center);
		rectangle(frame, Size(targets[j].x + targets[j].width, targets[j].y + targets[j].height),
			Point(targets[j].x, targets[j].y), Scalar(0, 255, 0), 1, 8, 0); // draw target
		if(i == 1) { // only 1 target is tracked
			pos.push_back(centers[j]);
			target = j;
		}
	}
	if (i > 1 && !pos.empty()) { // if more than a target is tracked insert the more probable center in trajectory
		Point last_tr = Point(pos[pos.size() - 1].x, pos[pos.size() - 1].y);
		double dist = norm(last_tr - centers[0]); // distance between latest tr and centers tracked
		size_t index = 0;
		for (size_t j = 1; j < i; j++) {
			if (norm(last_tr - centers[j]) < dist) {
				dist = norm(last_tr - centers[j]);
				index = j;
			}
		}
		pos.push_back(centers[index]);
		target = index;
	}
	if (fail_counter <= 15) {
		for (size_t j = 1; j < pos.size(); j++) {
			line(frame, pos[j], pos[j - 1], Scalar(0, 255, 150), 2, 8, 0); // draw trajectory
		}
	}
	else {
		pos.clear();
	}
	ostringstream dst;
	//calculate distance;
	if (!targets.empty()) {
		double distance = (focal_length * r_width) / (targets[target].width * 10);
		dst << "Distance: " << distance << "cm";
	}
	// write frame data
	ostringstream target_position;
	Scalar info_color;
	if (!pos.empty() && i != 0) {
		info_color = Scalar(0, 255, 255);
		target_position << "Position:{x = " << pos.back().x << "; y = " << pos.back().y << "}";
	}
	else {
		info_color = Scalar(0, 0, 255);
		target_position << "No target is tracked.";
	}
	putText(frame, target_position.str(),
		Point2f(10, 65), FONT_HERSHEY_DUPLEX, 0.45, info_color); // position info
	putText(frame, dst.str(),
		Point2f(10, 95), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 255, 0)); // position info
	return frame;
}