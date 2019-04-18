//
// Created by bogonets on 19. 4. 18.
//

#include <Yolov3TensorRT.hpp>
#include <utils.hpp>


Yolov3TensorRT::Yolov3TensorRT(std::string const & model_path, float nms_threshold)
    : _model_path(model_path), _nms_threshold(nms_threshold)
{
}

Yolov3TensorRT::~Yolov3TensorRT()
{
    // EMPTY.
}

std::vector<Yolov3TensorRT::Bbox> Yolov3TensorRT::predict(cv::Mat const & image)
{
    // Prepare image.
//    auto input = prepareImage(image);

    return std::vector<Bbox>();
}
