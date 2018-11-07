#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
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
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(48, 48) );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        Mat faceROI = frame_gray( faces[i] );
    }
	return frame;
}