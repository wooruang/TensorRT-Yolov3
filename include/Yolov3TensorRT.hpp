//
// Created by wooruang on 19. 4. 17.
//

#ifndef TENSORRTYOLOV3_YOLOV3TENSORRT_HPP
#define TENSORRTYOLOV3_YOLOV3TENSORRT_HPP

#include <opencv2/opencv.hpp>

#include <TrtNet.h>
#include <dataReader.h>


class Yolov3TensorRT
{
public:
    using Bbox = Tn::Bbox;
    using trtNet = Tn::trtNet;
    using RUN_MODE = Tn::RUN_MODE;
    using Detection = Yolo::Detection;

public:
    using SharedTrtNet = std::shared_ptr<trtNet>;

private:
    std::string _model_path;
    int _net_width;
    int _net_height;
    int _channel;
    float _nms_threshold;
    int _num_classes;

private:
    SharedTrtNet _net;

public:
    Yolov3TensorRT(std::string const & model_path, int net_width, int net_height, int channel, float nms_threshold, int num_classes);
    ~Yolov3TensorRT();

public:
    void init();
    void init(std::string const & prototxt, std::string const & caffemodel, std::vector<std::string> const & outputNodesName,
              std::vector<std::vector<float>> const & calibratorData, RUN_MODE mode);

public:
    std::vector<Bbox> predict(cv::Mat const & image);

};


#endif //TENSORRTYOLOV3_YOLOV3TENSORRT_HPP
