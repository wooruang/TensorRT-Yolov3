//
// Created by wooruang on 19. 4. 17.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <TensorRT/Yolov3TensorRT.hpp>


PYBIND11_MODULE(libTensorRTYolov3, m)
{
    namespace py = pybind11;

    m.doc() = "Yolov3 package";

    using Yolov3TensorRT = yolov3trt::Yolov3TensorRT;
    py::class_<Yolov3TensorRT>(m, "Yolov3TensorRT", py::buffer_protocol())
        .def(py::init<const std::string  &, int, int, int, float, int>())
        .def("init", &Yolov3TensorRT::init)
        .def("predict", &Yolov3TensorRT::predict)
        .def("predictFromPython", &Yolov3TensorRT::predictFromPython);
}
