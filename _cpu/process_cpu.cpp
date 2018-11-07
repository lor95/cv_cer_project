#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "process_cpu.h"

using namespace std;
using namespace cv;
CascadeClassifier cascade;

Mat process_cpu( Mat mainframe, CascadeClassifier target_cascade);
Mat detect_target( Mat frame );

Mat process_cpu( Mat mainframe, CascadeClassifier target_cascade) {
	cascade = target_cascade;
	return detect_target ( mainframe );
}
Mat detect_target( Mat frame) { 
    std::vector<Rect> targets;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect targets
    cascade.detectMultiScale( frame_gray, targets, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
	size_t i = targets.size();
	if (i != 1) {
		putText(frame, "RETRY", Point2f(25, 30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 0, 255, 255));
	}
    for ( i = 0; i < targets.size(); i++ )
    {
        Point center( targets[i].x + targets[i].width/2, targets[i].y + targets[i].height/2 );
        ellipse( frame, center, Size( targets[i].width/2, targets[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Mat faceROI = frame_gray( targets[i] );
    }
	return frame;
}