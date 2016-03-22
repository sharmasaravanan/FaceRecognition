#include "CAFFE_classifier.h"
using namespace cv;
using namespace std;
using namespace caffe;

// ----------------------------------------------------------------------------------------------------------
// boxInRange
// ----------------------------------------------------------------------------------------------------------



// netArchitecture = "eyes_test.prototxt"
// netWeights = "eyes_iter_100000"

CAFFE_classifier::CAFFE_classifier(string netArchitecture, string netWeights, string mean_image_filename,int _n_classes, bool useGPU)
{
    n_classes=_n_classes;
    
    if (useGPU)
    {
        Caffe::set_mode(Caffe::GPU);
        int device_id = 0;
        Caffe::SetDevice(device_id);
        cout << "Using GPU #" << device_id << endl;
    }
    else
    {
        cout << "Using CPU" << endl;
        Caffe::set_mode(Caffe::CPU);
    }
    //Caffe::set_phase(Caffe::TEST);
    net = new caffe::Net<float>(netArchitecture, caffe::TEST);
    net->CopyTrainedLayersFrom(netWeights);

    if (!mean_image_filename.empty())
    {
        mean_image = imread(mean_image_filename, 0);
        mean_image.convertTo(mean_image, CV_32FC1);       
        //imshow("mean",mean_image/255);
       //waitKey(10);
    }
}

CAFFE_classifier::~CAFFE_classifier(void)
{
    delete net;
}

void CAFFE_classifier::imageToDataLayer_gray(Mat& img, float*& data)
{
    int plane_sz = img.rows * img.cols;
    data = new float[plane_sz];
    memcpy((float*) data, (float*) img.data, plane_sz * sizeof (float));
}

void CAFFE_classifier::dataLayerToImage_gray(const float*& data, Mat& img)
{
    int plane_sz = img.rows * img.cols;
    memcpy((float*) img.data, (float*) data, plane_sz * sizeof (float));
}

void CAFFE_classifier::imageToDataLayer(Mat& img, float*& data)
{
    vector<Mat> ch;
    split(img, ch);
    int plane_sz = img.rows * img.cols;
    data = new float[plane_sz * 3];
    memcpy((float*) data, (float*) ch[0].data, plane_sz * sizeof (float));
    memcpy((float*) (data + plane_sz), (float*) ch[1].data, plane_sz * sizeof (float));
    memcpy((float*) (data + 2 * plane_sz), (float*) ch[2].data, plane_sz * sizeof (float));
}

void CAFFE_classifier::dataLayerToImage(const float*& data, Mat& img)
{
    vector<Mat> ch;
    split(img, ch);
    int plane_sz = img.rows * img.cols;
    memcpy((float*) ch[0].data, (float*) data, plane_sz * sizeof (float));
    memcpy((float*) ch[1].data, (float*) (data + plane_sz), plane_sz * sizeof (float));
    memcpy((float*) ch[2].data, (float*) (data + 2 * plane_sz), plane_sz * sizeof (float));
    merge(ch, img);
}

void CAFFE_classifier::classify(ImageSample& sample)
{
    float pain_prob, no_pain_prob;
    boost::shared_ptr<caffe::Layer<float> > netLayer = net->layer_by_name("data");
    float label = 0;
    Mat tmp = sample.scaled_image.clone();

    cvtColor(tmp,tmp,COLOR_BGR2GRAY);
    tmp.convertTo(tmp, CV_32FC1);
    
    if(!mean_image.empty())
    {
        tmp-=mean_image;
    }
    
  //  tmp.convertTo(tmp, CV_32FC3, 1.0 / 255.0);

    float* data;
    imageToDataLayer_gray(tmp, data);
cout << "Classifier " << endl;
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(netLayer)->Reset((float*) data, &label, 1);
cout << "Classifier 1" << endl;
    float loss = 0.0;
    vector<Blob<float>*> results = net->ForwardPrefilled(&loss);
cout << "Classifier 2" << endl;
    // Get probabilities
    const boost::shared_ptr<Blob<float> >& probLayer = net->blob_by_name("prob");
    cout << "Classifier 3 " << endl;
    const float* probs_out = probLayer->cpu_data();

    float m=-10000;
    for(int i=0;i<n_classes;++i)
    {
    sample.probabilities[i]=probs_out[i]; 
    if(m<probs_out[i])
    {
        sample.max_class=i;
        m=probs_out[i];
    }
    }
    
    delete data;
}
