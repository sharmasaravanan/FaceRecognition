///* 
// * File:   EyesDetector.h
// * Author: andrey
// *
// * Created on April 17, 2015, 2:41 PM
// */
//#include "ImageSample.h"
//#include <fstream>
//#include <sstream>
//
//#ifndef EYESDETECTOR_H
//#define	EYESDETECTOR_H
//
//class faceDetector
//{
//public:
//    cv::CascadeClassifier cascade;
//    std::vector<cv::Rect> rects;
//    faceDetector();
//    ~faceDetector();
//    void getFace(cv::Mat& img, cv::RotatedRect& faceRect);
//private:
//};
//
//#endif	/* EYESDETECTOR_H */
//

#define USE_DLIB_FD

#include "ImageSample.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <random>
#ifdef USE_DLIB_FD
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#endif

#ifndef EYESDETECTOR_H
#define	EYESDETECTOR_H

class faceDetector
{
public:
#ifndef USE_DLIB_FD
    cv::CascadeClassifier cascade;
#else
    dlib::frontal_face_detector detector;
#endif
    std::vector<cv::Rect> rects;
    faceDetector();
    ~faceDetector();
    void getFace(cv::Mat& img, cv::RotatedRect& faceRect);
private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
#ifdef USE_DLIB_FD
cv::Rect getRect(dlib::rectangle dlibrect);
#endif
};

#endif	/* EYESDETECTOR_H */
