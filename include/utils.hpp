//
// Created by bogonets on 19. 4. 18.
//

#ifndef TENSORRTYOLOV3_UTILS_HPP
#define TENSORRTYOLOV3_UTILS_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include <YoloLayer.h>
#include <dataReader.h>
#include <


std::vector<float> prepareImage(cv::Mat & img);

void DoNms(std::vector<Yolo::Detection> & detections, int classes, float nmsThresh);

std::vector<Tn::Bbox> postProcessImg(cv::Mat & img, std::vector<Yolo::Detection> & detections, int classes);

#endif //TENSORRTYOLOV3_UTILS_HPP
