/* 
 * File:   ImageSample.h
 * Author: andrey
 *
 * Created on April 17, 2015, 3:32 PM
 */

#ifndef IMAGESAMPLE_H
#define	IMAGESAMPLE_H

class ImageSample {
public:
    // Fields
    float max_class=0;
    float *probabilities;
    
    cv::RotatedRect location;
    cv::Mat scaled_image;
    // Methods
    ImageSample(int _n_classes);
    int n_classes;
    ~ImageSample();

    void setSample(cv::Mat& srcImg, cv::RotatedRect& location,int w,int h);
    void drawRoiRectangle(cv::Mat& img);
    void drawRoiRectangle(cv::Mat& img, int _state);
    bool boxInRange(cv::Mat& img, cv::RotatedRect& r);
    
private:
    void getQuadrangleSubPix_8u32f_CnR(const uchar* src, size_t src_step, cv::Size src_size, float* dst, size_t dst_step, cv::Size win_size, const double *matrix, int cn);
    void myGetQuadrangleSubPix(const cv::Mat& src, cv::Mat& dst, cv::Mat& m);
    void getRotRectImg(cv::RotatedRect rr, cv::Mat &img, cv::Mat& dst);
};

#endif	/* IMAGESAMPLE_H */

