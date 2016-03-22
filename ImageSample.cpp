/* 
 * File:   ImageSample.cpp
 * Author: andrey
 * 
 * Created on April 17, 2015, 3:32 PM
 */
#include "opencv2/opencv.hpp"
#include "ImageSample.h"

using namespace std;
using namespace cv;

//----------------------------------------------
// 
//----------------------------------------------

void ImageSample::getQuadrangleSubPix_8u32f_CnR(const uchar* src, size_t src_step, cv::Size src_size,
                                                float* dst, size_t dst_step, Size win_size,
                                                const double *matrix, int cn)
{
    int x, y, k;
    double A11 = matrix[0], A12 = matrix[1], A13 = matrix[2];
    double A21 = matrix[3], A22 = matrix[4], A23 = matrix[5];

    src_step /= sizeof (src[0]);
    dst_step /= sizeof (dst[0]);

    for (y = 0; y < win_size.height; y++, dst += dst_step)
    {
        double xs = A12 * y + A13;
        double ys = A22 * y + A23;
        double xe = A11 * (win_size.width - 1) + A12 * y + A13;
        double ye = A21 * (win_size.width - 1) + A22 * y + A23;

        if ((unsigned) (cvFloor(xs) - 1) < (unsigned) (src_size.width - 3) &&
                (unsigned) (cvFloor(ys) - 1) < (unsigned) (src_size.height - 3) &&
                (unsigned) (cvFloor(xe) - 1) < (unsigned) (src_size.width - 3) &&
                (unsigned) (cvFloor(ye) - 1) < (unsigned) (src_size.height - 3))
        {
            for (x = 0; x < win_size.width; x++)
            {
                int ixs = cvFloor(xs);
                int iys = cvFloor(ys);
                const uchar *ptr = src + src_step*iys;
                float a = (float) (xs - ixs), b = (float) (ys - iys), a1 = 1.f - a, b1 = 1.f - b;
                float w00 = a1*b1, w01 = a*b1, w10 = a1*b, w11 = a*b;
                xs += A11;
                ys += A21;

                if (cn == 1)
                {
                    ptr += ixs;
                    dst[x] = ptr[0] * w00 + ptr[1] * w01 + ptr[src_step] * w10 + ptr[src_step + 1] * w11;
                }
                else if (cn == 3)
                {
                    ptr += ixs * 3;
                    float t0 = ptr[0] * w00 + ptr[3] * w01 + ptr[src_step] * w10 + ptr[src_step + 3] * w11;
                    float t1 = ptr[1] * w00 + ptr[4] * w01 + ptr[src_step + 1] * w10 + ptr[src_step + 4] * w11;
                    float t2 = ptr[2] * w00 + ptr[5] * w01 + ptr[src_step + 2] * w10 + ptr[src_step + 5] * w11;

                    dst[x * 3] = t0;
                    dst[x * 3 + 1] = t1;
                    dst[x * 3 + 2] = t2;
                }
                else
                {
                    ptr += ixs*cn;
                    for (k = 0; k < cn; k++)
                        dst[x * cn + k] = ptr[k] * w00 + ptr[k + cn] * w01 +
                            ptr[src_step + k] * w10 + ptr[src_step + k + cn] * w11;
                }
            }
        }
        else
        {
            for (x = 0; x < win_size.width; x++)
            {
                int ixs = cvFloor(xs), iys = cvFloor(ys);
                float a = (float) (xs - ixs), b = (float) (ys - iys), a1 = 1.f - a, b1 = 1.f - b;
                float w00 = a1*b1, w01 = a*b1, w10 = a1*b, w11 = a*b;
                const uchar *ptr0, *ptr1;
                xs += A11;
                ys += A21;

                if ((unsigned) iys < (unsigned) (src_size.height - 1))
                    ptr0 = src + src_step * iys, ptr1 = ptr0 + src_step;
                else
                    ptr0 = ptr1 = src + (iys < 0 ? 0 : src_size.height - 1) * src_step;

                if ((unsigned) ixs < (unsigned) (src_size.width - 1))
                {
                    ptr0 += ixs*cn;
                    ptr1 += ixs*cn;
                    for (k = 0; k < cn; k++)
                        dst[x * cn + k] = ptr0[k] * w00 + ptr0[k + cn] * w01 + ptr1[k] * w10 + ptr1[k + cn] * w11;
                }
                else
                {
                    ixs = ixs < 0 ? 0 : src_size.width - 1;
                    ptr0 += ixs*cn;
                    ptr1 += ixs*cn;
                    for (k = 0; k < cn; k++)
                        dst[x * cn + k] = ptr0[k] * b1 + ptr1[k] * b;
                }
            }
        }
    }
}


//----------------------------------------------
// 
//----------------------------------------------

void ImageSample::myGetQuadrangleSubPix(const Mat& src, Mat& dst, Mat& m)
{
    CV_Assert(src.channels() == dst.channels());

    cv::Size win_size = dst.size();
    double matrix[6];
    cv::Mat M(2, 3, CV_64F, matrix);
    m.convertTo(M, CV_64F);
    double dx = (win_size.width - 1)*0.5;
    double dy = (win_size.height - 1)*0.5;
    matrix[2] -= matrix[0] * dx + matrix[1] * dy;
    matrix[5] -= matrix[3] * dx + matrix[4] * dy;

    if (src.depth() == CV_8U && dst.depth() == CV_32F)
        getQuadrangleSubPix_8u32f_CnR(src.data, src.step, src.size(),
                                      (float*) dst.data, dst.step, dst.size(),
                                      matrix, src.channels());
    else
    {
        CV_Assert(src.depth() == dst.depth());
        cv::warpAffine(src, dst, M, dst.size(),
                       cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                       cv::BORDER_REPLICATE);
    }
}
//----------------------------------------------------------
// 
//----------------------------------------------------------

void ImageSample::getRotRectImg(cv::RotatedRect rr, Mat &img, Mat& dst)
{
    Mat m(2, 3, CV_64FC1);
    float ang = rr.angle * CV_PI / 180.0;
    m.at<double>(0, 0) = cos(ang);
    m.at<double>(1, 0) = sin(ang);
    m.at<double>(0, 1) = -sin(ang);
    m.at<double>(1, 1) = cos(ang);
    m.at<double>(0, 2) = rr.center.x;
    m.at<double>(1, 2) = rr.center.y;
    myGetQuadrangleSubPix(img, dst, m);
}
//----------------------------------------------
// 
//----------------------------------------------

bool ImageSample::boxInRange(cv::Mat& img, cv::RotatedRect& r)
{
    Point2f rect_points[4];
    r.points(rect_points);

    cv::Rect img_r = cv::Rect(0, 0, img.cols - 1, img.rows - 1);

    bool result = true;
    for (int i = 0; i < 4; ++i)
    {
        if (!img_r.contains(rect_points[i]))
        {
            result = false;
            break;
        }
    }
    return result;
}

ImageSample::ImageSample(int _n_classes)
{
    int n_classes = _n_classes;
    location = cv::RotatedRect();
    scaled_image = cv::Mat();

    probabilities = new float[n_classes];

    for (int i = 0; i < n_classes; ++i)
    {
        probabilities[i] = 0;
    }

}

ImageSample::~ImageSample()
{
    delete probabilities;
}

void ImageSample::setSample(cv::Mat& srcImg, cv::RotatedRect& _location, int sample_width, int sample_height)
{
    if (boxInRange(srcImg, _location) && _location.size.area() > 10)
    {
        Mat tmp = Mat(_location.size.height, _location.size.width, CV_8UC3);
        getRotRectImg(_location, srcImg, tmp);
        cv::resize(tmp, scaled_image, cv::Size(sample_width, sample_height));
        cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
        this->location = _location;
    }
    else
    {
        location = cv::RotatedRect();
        scaled_image.release();
    }

}

void ImageSample::drawRoiRectangle(cv::Mat& img)
{
    cv::Scalar state_colors[] = {cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};

    // rotated rectangle
    Point2f rect_points[4];
    location.points(rect_points);
    cv::Rect faces=location.boundingRect();
     int p_x_1 = faces.x;
        int p_x_2 = faces.x + faces.width*0.25;
        int p_x_3 = faces.x + faces.width * (1 - 0.25);
        int p_x_4 = faces.x + faces.width;

        int p_y_1 = faces.y;
        int p_y_2 = faces.y + faces.height*0.25;
        int p_y_3 = faces.y + faces.height * (1 - 0.25);
        int p_y_4 = faces.y + faces.height;
        int state=0;
        line(img, Point(p_x_1, p_y_1), Point(p_x_2, p_y_1), state_colors[state], 2);
        line(img, Point(p_x_3, p_y_1), Point(p_x_4, p_y_1), state_colors[state], 2);
        line(img, Point(p_x_4, p_y_1), Point(p_x_4, p_y_2), state_colors[state], 2);
        line(img, Point(p_x_4, p_y_3), Point(p_x_4, p_y_4), state_colors[state], 2);
        line(img, Point(p_x_4, p_y_4), Point(p_x_3, p_y_4), state_colors[state], 2);
        line(img, Point(p_x_2, p_y_4), Point(p_x_1, p_y_4), state_colors[state], 2);
        line(img, Point(p_x_1, p_y_4), Point(p_x_1, p_y_3), state_colors[state], 2);
        line(img, Point(p_x_1, p_y_2), Point(p_x_1, p_y_1), state_colors[state], 2);
//    for (int j = 0; j < 4; j++)
//    {
//        line(img, rect_points[j], rect_points[(j + 1) % 4], state_colors[0], 1, 8);
//    }

}

void ImageSample::drawRoiRectangle(cv::Mat& img, int _state)
{
    cv::Scalar state_colors[] = {cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    // rotated rectangle
    Point2f rect_points[4];
    location.points(rect_points);
    for (int j = 0; j < 4; j++)
    {
        line(img, rect_points[j], rect_points[(j + 1) % 4], state_colors[_state], 1, 8);
    }
}
