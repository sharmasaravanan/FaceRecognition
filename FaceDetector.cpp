///* 
// * File:   EyesDetector.cpp
// * Author: andrey
// * 
// * Created on April 17, 2015, 2:41 PM
// */
//#include <opencv2/opencv.hpp>
//#include <FaceDetector.h>
//
//using namespace cv;
//using namespace std;
//// -------------------------------------------------------------------------
//// Create Eyes detector.
//// First parameter is path to program executable.
//// Second parameter is the videocapture address.
//// -------------------------------------------------------------------------
//
//faceDetector::faceDetector()
//{
//    cascade.load("haarcascade_frontalface_alt2.xml");
//}
//// -------------------------------------------------------------------------
//// Destructor
//// -------------------------------------------------------------------------
//
//faceDetector::~faceDetector()
//{
//}
//// -------------------------------------------------------------------------
//// Compute eyes regions and put the result to RotatedRect for each eye.
//// -------------------------------------------------------------------------
//
//void faceDetector::getFace(cv::Mat& image, cv::RotatedRect& face)
//{
//
//    std::vector<cv::Rect> res;
//    cv::Mat im(image.size(), CV_8UC1);
//    if (image.channels() == 3)
//    {
//        cvtColor(image, im, cv::COLOR_BGR2GRAY);
//    }
//    else
//    {
//        image.copyTo(im);
//    }
//
//    cascade.detectMultiScale(im, res, 1.1, 2, CASCADE_FIND_BIGGEST_OBJECT, Size(image.cols / 3, image.rows / 3));
//
//    if (res.size() > 0)
//    {
//        // Fill rotated rectangle structure
//        face.center = (res[0].tl() + res[0].br()) / 2;
//        face.angle = 0;
//        face.size.width = res[0].width;
//        face.size.height = res[0].height;
//    }
//    else
//    {
//      face.center = cv::Point(0,0);
//        face.angle = 0;
//        face.size.width = 0;
//        face.size.height = 0;  
//    }
//
//
//}

/* 
 * File:   EyesDetector.cpp
 * Author: andrey
 * 
 * Created on April 17, 2015, 2:41 PM
 */
#include <opencv2/opencv.hpp>
#include "FaceDetector.h"

using namespace cv;
using namespace std;
#ifdef USE_DLIB_FD

cv::Rect faceDetector::getRect(dlib::rectangle dlibrect)
    {
    cv::Rect r(dlibrect.left(), dlibrect.bottom() - dlibrect.height(), dlibrect.width(), dlibrect.height());
    return r;
    }
#endif
// -------------------------------------------------------------------------
// Create Eyes detector.
// First parameter is path to program executable.
// Second parameter is the videocapture address.
// -------------------------------------------------------------------------

faceDetector::faceDetector()
    {
#ifdef USE_DLIB_FD
    detector = dlib::get_frontal_face_detector();
#else
    cascade.load("./../../../../Model/haarcascade_frontalface_alt2.xml");
#endif
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = std::default_random_engine(seed);
    distribution = std::uniform_int_distribution<int>(-15, 15);
    }
// -------------------------------------------------------------------------
// Destructor
// -------------------------------------------------------------------------

faceDetector::~faceDetector()
    {
    }
// -------------------------------------------------------------------------
// Compute eyes regions and put the result to RotatedRect for each eye.
// -------------------------------------------------------------------------

void transformFaceRect(cv::Rect& r, float dx, float dy, float sx, float sy)
    {
    float w = r.width;
    float h = r.height;
    float x = r.x;
    float y = r.y;
    float cx = x + w / 2.0;
    float cy = y + h / 2.0;
    float new_w = w*sx;
    float new_h = h*sy;
    float new_x = cx - new_w / 2.0 + dx*new_w;
    float new_y = cy - new_h / 2.0 + dy*new_h;
    r = cv::Rect(new_x, new_y, new_w, new_h);
    }

void faceDetector::getFace(cv::Mat& image, cv::RotatedRect& face)
    {

    std::vector<cv::Rect> res;
#ifndef USE_DLIB_FD
    cv::Mat im(image.size(), CV_8UC1);
    if (image.channels() == 3)
        {
        cvtColor(image, im, cv::COLOR_BGR2GRAY);
        }
    else
        {
        image.copyTo(im);
        }

    cascade.detectMultiScale(im, res, 1.1, 2, CASCADE_FIND_BIGGEST_OBJECT, Size(30, 30));
    if (res.size() > 0)
        {
        transformFaceRect(res[0], 0, 0.0, 1.1, 1.1);
        cv::Rect roi(0, 0, im.cols, im.rows);
        if ((res[0] & roi) != res[0]) // Rectangles intersection
            {
            res.clear();
            }
        }
#else
    cv::Mat im(image.size(), CV_8UC3);
    if (image.channels() == 3)
        {
        image.copyTo(im);
        }
    else
        {
        cvtColor(image, im, cv::COLOR_GRAY2BGR);
        }
    dlib::cv_image<dlib::bgr_pixel> cimg(im);
    // Detect faces 
    std::vector<dlib::rectangle> faces = detector(cimg);
    if (faces.size() > 0)
        {
        for (int i = 0; i < 1/*faces.size()*/; ++i)
            {

            if (faces[i].bl_corner().x() > 0 &&
                faces[i].bl_corner().y() < image.rows &&
                faces[i].tr_corner().x() < image.cols &&
                faces[i].tr_corner().y() > 0)
                {
                res.push_back(getRect(faces[i]));
                }
            }
        for (int i = 0; i < 1/*faces.size()*/; ++i)
            {
            // cv::rectangle(image, getRect(faces[i]), Scalar(0, 255, 0), 1);
            }
        faces.clear();
        }
#endif

    if (res.size() > 0)
        {
        // Fill rotated rectangle structure
        face.center = (res[0].tl() + res[0].br()) * 0.5;
        face.angle = 0;//distribution(generator);
        face.size.width = res[0].width;
        face.size.height = res[0].height;
        }
    else
        {
        face.center = Point(0, 0);
        face.angle = 0;
        face.size.height = 0;
        face.size.width = 0;
        }

    }