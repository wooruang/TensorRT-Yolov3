//
// Created by wooruang on 19. 4. 18.
//

#include <TensorRT/utils.hpp>

#include <chrono>

std::vector<float> prepareImage(cv::Mat const & img, int channel, int net_width, int net_height)
{
    float scale = std::min(float(net_width)/img.cols,float(net_height)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, cv::INTER_CUBIC);

    cv::Mat cropped(net_height, net_width,CV_8UC3, 127);
    cv::Rect rect((net_width - scaleSize.width)/2, (net_height-scaleSize.height)/2, scaleSize.width,scaleSize.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (channel == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(img_float, input_channels);

    std::vector<float> result(net_height*net_width*channel);
    auto data = result.data();
    int channelLength = net_height * net_width;
    for (int i = 0; i < channel; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void DoNms(std::vector<Yolo::Detection> & detections, int classes, float nmsThresh)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<Yolo::Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    std::vector<Yolo::Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i];
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](Yolo::Detection const & left, Yolo::Detection const & right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Time taken for nms is " << total << " ms." << std::endl;
}

void postProcessImg(cv::Mat const & img, std::vector<Yolo::Detection> & detections,
                                     int classes, float nms_threshold, int net_width, int net_height)
{

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = std::min(float(net_width) / width, float(net_height) / height);
    float scaleSize[] = {width * scale, height * scale};

    //correct box
    for (auto & item : detections) {
        auto & bbox = item.bbox;
        bbox[0] = (bbox[0] * net_width - (net_width - scaleSize[0]) / 2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * net_height - (net_height - scaleSize[1]) / 2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    if (nms_threshold > 0)
        DoNms(detections, classes, nms_threshold);
}

std::vector<Tn::Bbox> toBbox(cv::Mat const & img, std::vector<Yolo::Detection> & detections)
{
    int width = img.cols;
    int height = img.rows;

    std::vector<Tn::Bbox> boxes;
    for(auto const & item : detections)
    {
        auto & b = item.bbox;
        Tn::Bbox bbox =
            {
                item.classId,   //classId
                std::max(int((b[0]-b[2]/2.)*width),0), //left
                std::min(int((b[0]+b[2]/2.)*width),width), //right
                std::max(int((b[1]-b[3]/2.)*height),0), //top
                std::min(int((b[1]+b[3]/2.)*height),height), //bot
                item.prob       //score
            };
        boxes.push_back(bbox);
    }

    return boxes;
}

std::vector<std::vector<float>> toVector(cv::Mat const & img, std::vector<Yolo::Detection> & detections)
{
    int width = img.cols;
    int height = img.rows;

    std::vector<std::vector<float>> boxes;
    for(auto const & item : detections)
    {
        auto & b = item.bbox;
        std::vector<float> bbox =
            {
                static_cast<float>(item.classId),   //classId
                static_cast<float>(std::max((b[0]-b[2]/2.)*width,0.)), //left
                static_cast<float>(std::min((b[0]+b[2]/2.)*width, static_cast<double>(width))), //right
                static_cast<float>(std::max((b[1]-b[3]/2.)*height, 0.)), //top
                static_cast<float>(std::min((b[1]+b[3]/2.)*height, static_cast<double>(height))), //bot
                item.prob       //score
            };
        boxes.push_back(bbox);
    }

    return boxes;
}

std::vector<Tn::Bbox> postProcessImgToBbox(cv::Mat const & img, std::vector<Yolo::Detection> & detections,
                          int classes, float nms_threshold, int net_width, int net_height)
{
    postProcessImg(img, detections, classes, nms_threshold, net_width, net_height);
    return toBbox(img, detections);
}

std::vector<std::vector<float>> postProcessImgToVector(cv::Mat const & img, std::vector<Yolo::Detection> & detections,
                            int classes, float nms_threshold, int net_width, int net_height)
{
    postProcessImg(img, detections, classes, nms_threshold, net_width, net_height);
    return toVector(img, detections);
}
