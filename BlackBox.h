#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/opencv.hpp"
#include "time.h"
#include "CAFFE_classifier.h"
#include "fstream"
#include <numeric>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "FaceDetector.h"
#ifndef BLACKBOX_H
#define	BLACKBOX_H

class BlackBox
{
public:
    BlackBox(std::string path,
             cv::VideoCapture* _video_capture,
             std::string netArchitecture,
             std::string netWeights,
             std::string mean_image_filename,         
             bool useGPU = false);
    void setMode(int mode);
    void BlackBox::drawProbabilities(cv::Mat& frame,ImageSample& faceSample);
    void process(cv::Mat& frame);
    void BlackBox::FaceCropping(cv::Mat& frame,std::string path);
    std::vector<std::string> class_names;
     ImageSample* faceSample;
     int count=0;
       int count1=0;
    ~BlackBox();
private:

    

    float* avg_emotions;
    // Eyes regions extractor
    faceDetector* FDet;
    // Neural network classifier for eyes
    CAFFE_classifier* cc;
    // Eyes structure for feedeng to neural network
   
    // Eyes regions
    cv::RotatedRect faceRect;
    // 0 - normal
    // 1 - smile
    // 2 - no smile
    int _mode;
};

#endif	/* BLACKBOX_H */

