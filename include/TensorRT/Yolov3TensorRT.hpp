//
// Created by wooruang on 19. 4. 17.
//

#ifndef TENSORRTYOLOV3_YOLOV3TENSORRT_HPP
#define TENSORRTYOLOV3_YOLOV3TENSORRT_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <TrtNet.h>
#include <TensorRT/dataReader.h>

namespace yolov3trt {

class Yolov3TensorRT
{
public:
    using Bbox = Tn::Bbox;
    using trtNet = Tn::trtNet;
    using RUN_MODE = Tn::RUN_MODE;
    using Detection = Yolo::Detection;

public:
    using SharedTrtNet = std::shared_ptr<trtNet>;

public:
    using PyArrary = pybind11::array;
    using PyArraryT = pybind11::array_t<uint8_t, PyArrary::c_style | PyArrary::forcecast>;

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
    explicit Yolov3TensorRT(std::string const & model_path, int net_width, int net_height, int channel, float nms_threshold, int num_classes);
    ~Yolov3TensorRT();

public:
    void init();
    void initByPrototxt(std::string const & prototxt, std::string const & caffemodel, std::vector<std::string> const & outputNodesName,
              std::vector<std::vector<float>> const & calibratorData, RUN_MODE mode);

public:
    std::vector<std::vector<float>> predictFromPython(PyArraryT array, int width, int height, int channel);
    std::vector<Detection> predict(cv::Mat const & image);
    std::vector<Bbox> predictToBbox(cv::Mat const & image);
    std::vector<std::vector<float>> predictToVector(cv::Mat const & image);

public:
    static cv::Mat convertToMat(std::vector<uint8_t> const & pos, int width, int height, int channel)
    {
        int type = CV_MAKETYPE(CV_8U,channel);
        cv::Mat m(height, width, type);
        std::memcpy(m.data, pos.data(), width*height*channel);
        return m;
    };

};

} //namespace yolov3trt.

#endif //TENSORRTYOLOV3_YOLOV3TENSORRT_HPP
