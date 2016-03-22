#include "opencv2/opencv.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "ImageSample.h"
#pragma once

class CAFFE_classifier
{
private:
    // Caffe network instance
    caffe::Net<float>* net;
    // Mean image
    cv::Mat mean_image;
    // Load caffe data layer with opencv's image  (for gray images)
    void imageToDataLayer_gray(cv::Mat& img, float*& data);
    // Load image from caffe data layer to opencv's image (for gray images)
    void dataLayerToImage_gray(const float*& data, cv::Mat& img);
     // Load caffe data layer with opencv's image  (for gray images)
    void imageToDataLayer(cv::Mat& img, float*& data);
    // Load image from caffe data layer to opencv's image
    void dataLayerToImage(const float*& data, cv::Mat& img);
    
    int n_classes;
public:
    // Caffe network classifier constructor
    // example of usage: CAFFE_classifier cc("blink_deploy.prototxt", "blink.caffemodel","", false);
    CAFFE_classifier(std::string netArchitecture, std::string netWeights, std::string mean_image_filename,int _n_classes, bool useGPU = false);
    void classify(ImageSample& sample);
    ~CAFFE_classifier(void);
};

