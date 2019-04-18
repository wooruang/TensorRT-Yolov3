//
// Created by wooruang on 19. 4. 17.
//

#ifndef TENSORRTYOLOV3_YOLOV3TENSORRT_HPP
#define TENSORRTYOLOV3_YOLOV3TENSORRT_HPP

#include <opencv2/opencv.hpp>
#include <dataReader.h>


class Yolov3TensorRT
{
public:
    using Bbox = Tn::Bbox;

private:
    std::string _model_path;
    float _nms_threshold;

public:
    Yolov3TensorRT(std::string const & model_path, float nms_threshold);
    ~Yolov3TensorRT();

public:
    std::vector<Bbox> predict(cv::Mat const & image);

};


#endif //TENSORRTYOLOV3_YOLOV3TENSORRT_HPP
