// sudo sh -c 'echo "/home/andrey/caffe/build/lib" > /etc/ld.so.conf.d/caffe.conf'
// ldconfig
#include <chrono>
#include "BlackBox.h"
#include "clmtacker.h"

using namespace caffe;
using namespace std;
using namespace cv;
using namespace std::chrono;
// Main processing class
// It gets frame as input, detects eyes regions
// and recognizes the state of eyes (open/closed)

BlackBox::BlackBox(std::string path,
                   cv::VideoCapture* _video_capture,
                   std::string netArchitecture,
                   std::string netWeights,
                   std::string mean_image_filename,
                   bool useGPU)
{
    std::ifstream infile("./TestByBalaji/labels.txt");
    std::string line;
    while (std::getline(infile, line))
    {
        class_names.push_back(line);
    }
    
    // Caffe classifier
    cc = new CAFFE_classifier(netArchitecture, netWeights, mean_image_filename,class_names.size(), useGPU);
    FDet = new faceDetector();
    facecrop=new algobucket();
    _mode = 0;
    out.open("./TestByBalaji/result.txt",ios::out);
     outfile.open("./TestByBalaji/result_analysis.csv",ios::out);
    avg_emotions=new float[class_names.size()];
    faceSample=new ImageSample(class_names.size());
    for (int i = 0; i < class_names.size(); ++i)
    {
        avg_emotions[i] = 0;
    }
}
#define collect_emotions0
void BlackBox::setMode(int mode)
{
    _mode = mode;
   
}

void BlackBox::drawProbabilities(cv::Mat& frame, ImageSample& faceSample,string cel)
{
    float N = class_names.size(); // Averaging factor
    cout<<"##################################################"<<endl;
   
//#ifdef Parser
//#else
//    out.open("result");
//            static int countcheck=0;
//           
//        if(int(faceSample.max_class)==3)
//        {
//            countcheck++;
//        }
//        cout<<"result;"<<countcheck<<endl;
//#endif
         
    for (int i = 0; i < class_names.size(); ++i)
    {
        float len = faceSample.probabilities[i]*100;
        avg_emotions[i] = (N * avg_emotions[i] + len) / (N + 1.0);
        char avg[200];
        sprintf(avg,"%.4f",faceSample.probabilities[i]);
      //  cv::rectangle(frame, cv::Rect(30, i * 30, avg_emotions[i], 20), Scalar(0, 255, 0), -1);
    //    cv::putText(frame, avg, cv::Point(140, i * 30 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1, 8);
     //   cv::putText(frame, class_names[i], cv::Point(30, i * 30 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1, 8);
        //cv::rectangle(frame, Rect(0, i * 15, avg_emotions[i], 10), Scalar(0, 255, 0), -1);
        //cv::putText(frame, class_names[i], Point(0, i * 15 + 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 0, 0), 1, 8);
//        static int countcheck=0;
//        if(int(faceSample.max_class)==3)
//        {
//            countcheck++;
//        }
//        cout<<"resul;"
        
        if(cel==class_names[i])
        {
           //outfile <<"Correct Desion,"<<class_names[i]<<","<<faceSample.probabilities[i]<<endl; 
            writename=class_names[i];
           prob=faceSample.probabilities[i];
           filter=0;
           if(faceSample.probabilities[int(faceSample.max_class)]>=0.996)
           {
               int r=0;
               if(class_names[int(faceSample.max_class)]==cel)
               {
                   r=1;
               }
            outfile <<"Correct Desion,"<<writename<<","<<prob<<",Caffe Desion,"<<class_names[int(faceSample.max_class)]<<","<<faceSample.probabilities[int(faceSample.max_class)]<<","<<r<<endl;
           filter=1;
           }    
           }
        if(int(faceSample.max_class)==i)
        {
           // if(faceSample.probabilities[i]>0.99)
           // {
           // out<<"--------------------------------------------------"<<endl;
            cout<<"--------------------------------------------------"<<endl; 
            cout<<class_names[i]<<"="<<faceSample.probabilities[i]<<endl;
             out<<class_names[i]<<"="<<faceSample.probabilities[i]<<endl;
             
//             if(writename==class_names[i])
//             {
//                 
//             outfile <<"Correct Desion,"<<writename<<","<<prob<<",Caffe Desion,"<<class_names[i]<<","<<faceSample.probabilities[i]<<","<<"1"<<endl;
//             }
//             else
//             {
//                 
//               outfile <<"Correct Desion,"<<writename<<","<<prob<<",Caffe Desion,"<<class_names[i]<<","<<faceSample.probabilities[i]<<","<<"0"<<endl;
//             }
             cout<<"--------------------------------------------------"<<endl;
            // out<<"--------------------------------------------------"<<endl;
          //  }
          //  else
           // {
//                out<<"--------------------------------------------------"<<endl;
//            cout<<"--------------------------------------------------"<<endl; 
//            cout<<class_names[i]<<"="<<faceSample.probabilities[i]<<endl;
//             out<<class_names[i]<<"="<<faceSample.probabilities[i]<<endl;
//            cout<<"--------------------------------------------------"<<endl;
//             out<<"--------------------------------------------------"<<endl; 
           // }
            // cv::rectangle(frame, cv::Rect(30, 30, avg_emotions[i], 20), Scalar(0, 255, 0), -1);
        cv::putText(frame, avg, cv::Point(140, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1, 8);
        cv::putText(frame, class_names[i], cv::Point(30, 30 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1, 8);
        }
        else
        {
             //cout<<class_names[i]<<"="<<faceSample.probabilities[i]<<endl;
             // out<<class_names[i]<<"="<<faceSample.probabilities[i]<<endl;
        }
    }
   
}
void BlackBox::FaceCropping(cv::Mat& frame,string path)
{
    FDet->getFace(frame, faceRect);
   
    if (faceRect.size.area() > 10)
    {
        faceSample->setSample(frame, faceRect, 96, 96);

        if (faceSample->scaled_image.cols > 0 && faceSample->scaled_image.rows > 0)
        {
            imshow("face", faceSample->scaled_image);
            imwrite(path.c_str(),faceSample->scaled_image);
        }
        
    }
    else
    {
        if( remove( path.c_str()) != 0 )
    perror( "Error deleting file" );
  else
    puts( "File successfully deleted" );
    }
}
void BlackBox::process(cv::Mat& frame,string cel)
{
   //  cout<<"enter2"<<endl;
   // FDet->getFace(frame, faceRect);
    cv::Mat clmcropped;
    int ret=facecrop->clmfacecropping(frame,clmcropped);
    cout<<"return code "<<ret<<endl;
     imwrite("cropped.jpg",clmcropped);
    // cout<<"enter4"<<endl;
    count1=0;
   // clmcropped.s
    if (clmcropped.cols > 10 && clmcropped.rows > 10 && ret==1000)
    {
         cout<<"enter0"<<endl;
          cv::resize(clmcropped,clmcropped,cv::Size(64,64));
        faceSample->scaled_image=clmcropped;//(frame, faceRect, 112, 112);
       
       // cout<<"enter"<<endl;
        if (faceSample->scaled_image.cols > 0 && faceSample->scaled_image.rows > 0)
        {
            count1=1;
            count++;
             imshow("face", faceSample->scaled_image);
          //  imwrite(newpath,faceSample->scaled_image);
           //faceSample->scaled_image=frameone;
            cc->classify(*faceSample);
            faceSample->drawRoiRectangle(frame);
            drawProbabilities(frame, *faceSample,cel);
        }
          else
    {
        for (int i = 0; i < class_names.size(); ++i)
    {
        faceSample->probabilities[i]=0;
    }
    drawProbabilities(frame, *faceSample,cel);
    }
      
    }
    else
    {
        wrong_count++;
    }
#ifdef collect_emotions
    if (_mode > 0 && faceSample->max_class != _mode - 1) // shown pain but classified as no_pain
    {
        milliseconds ms = duration_cast< milliseconds >(high_resolution_clock::now().time_since_epoch());
        char buf[1024];
        sprintf(buf, "./%s/%lu.jpg", class_names[_mode - 1].c_str(), ms.count());
        cout << buf << endl;
        Mat tmp;
        cvtColor(faceSample->scaled_image, tmp, COLOR_BGR2GRAY);
        imwrite(buf, tmp);


        sprintf(buf, "./%s/%lu_flip.jpg", class_names[_mode - 1].c_str(), ms.count());
        cout << buf << endl;
        cvtColor(faceSample->scaled_image, tmp, COLOR_BGR2GRAY);
        cv::flip(tmp, tmp, 1);
        imwrite(buf, tmp);
    }


    if (_mode > 0)
    {
        cv::putText(frame, "Current mode:" + class_names[_mode - 1], Point(0, frame.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);
    }
    else
    {
        cv::putText(frame, "Current mode: normal", Point(0, frame.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 2, 8);
    }
    #endif
}

BlackBox::~BlackBox()
{
    delete cc;
    delete FDet;
    delete avg_emotions;
    delete faceSample;
     out.close();
}


