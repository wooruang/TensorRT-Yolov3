//
// Created by wooruang on 19. 4. 18.
//

#include <TensorRT/Yolov3TensorRT.hpp>
#include <TensorRT/utils.hpp>

namespace yolov3trt {

Yolov3TensorRT::Yolov3TensorRT(std::string const & model_path,
                               int net_width, int net_height, int channel,
                               float nms_threshold, int num_classes)
    : _model_path(model_path), _net_width(net_width), _net_height(net_height), _channel(channel),
    _nms_threshold(nms_threshold), _num_classes(num_classes)
{
    // EMPTY.
}

Yolov3TensorRT::~Yolov3TensorRT()
{
    // EMPTY.
}

void Yolov3TensorRT::init()
{
    _net.reset(new trtNet(_model_path));
}

void Yolov3TensorRT::initByPrototxt(std::string const & prototxt, std::string const & caffemodel,
                                    std::vector<std::string> const & outputNodesName,
                                    std::vector<std::vector<float>> const & calibratorData, Yolov3TensorRT::RUN_MODE mode)
{
    _net.reset(new trtNet(prototxt, caffemodel, outputNodesName, calibratorData, mode));
}

std::vector<std::vector<float>> Yolov3TensorRT::predictFromPython(PyArraryT array, int width, int height, int channel)
{
    // std::cout << "w: " << width << ", h: " << height << ", c: " << channel << std::endl;
    // check input dimensions
    if ( array.ndim()     != 3 )
        throw std::runtime_error("Input should be 2-D NumPy array");

    // allocate std::vector (to pass to the C++ function)
    std::vector<uint8_t> pos(array.size());

    // copy py::array -> std::vector
    std::memcpy(pos.data(),array.data(),array.size()*sizeof(uint8_t));

    cv::Mat t = convertToMat(pos, width, height, channel);

    auto bboxes = predictToVector(t);

    return bboxes;
}

std::vector<Yolov3TensorRT::Detection> Yolov3TensorRT::predict(cv::Mat const & image)
{
    if (_net == nullptr) {
        return std::vector<Detection>();

    }
    int outputCount = _net->getOutputSize()/sizeof(float);
    std::unique_ptr<float[]> outputData(new float[outputCount]);

    // Prepare image.
    std::vector<float> inputData = prepareImage(image, _channel, _net_width, _net_height);
    if (!inputData.data())
        return std::vector<Detection>();

    _net->doInference(inputData.data(), outputData.get());

    //Get Output
    auto output = outputData.get();

    //first detect count
    int count = output[0];
    //later detect result
    std::vector<Detection> result;
    result.resize(count);
    memcpy(result.data(), &output[1], count*sizeof(Detection));

    return result;
}

std::vector<Yolov3TensorRT::Bbox> Yolov3TensorRT::predictToBbox(cv::Mat const & image)
{
    auto result = predict(image);

    auto boxes = postProcessImgToBbox(image, result, _num_classes, _nms_threshold, _net_width, _net_height);

    return boxes;
}

std::vector<std::vector<float>> Yolov3TensorRT::predictToVector(cv::Mat const & image)
{
    auto result = predict(image);

    auto boxes = postProcessImgToVector(image, result, _num_classes, _nms_threshold, _net_width, _net_height);

    return boxes;
}

} //namespace yolov3trt.
