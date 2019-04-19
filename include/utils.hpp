//
// Created by wooruang on 19. 4. 18.
//

#ifndef TENSORRTYOLOV3_UTILS_HPP
#define TENSORRTYOLOV3_UTILS_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include <YoloLayer.h>
#include <dataReader.h>

std::vector<float> prepareImage(cv::Mat const & img, int channel, int net_width, int net_height);

void DoNms(std::vector<Yolo::Detection> & detections, int classes, float nmsThresh);

std::vector<Tn::Bbox> postProcessImg(cv::Mat const  & img, std::vector<Yolo::Detection> & detections,
                                     int classes, float nms_threshold, int net_width, int net_height);

#endif //TENSORRTYOLOV3_UTILS_HPP
