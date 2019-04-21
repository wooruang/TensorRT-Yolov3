//
// Created by wooruang on 19. 4. 18.
//

#ifndef TENSORRTYOLOV3_UTILS_HPP
#define TENSORRTYOLOV3_UTILS_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include <YoloLayer.h>
#include <TensorRT/dataReader.h>

std::vector<float> prepareImage(cv::Mat const & img, int channel, int net_width, int net_height);

void DoNms(std::vector<Yolo::Detection> & detections, int classes, float nmsThresh);

void postProcessImg(cv::Mat const  & img, std::vector<Yolo::Detection> & detections,
                                     int classes, float nms_threshold, int net_width, int net_height);

std::vector<Tn::Bbox> toBbox(cv::Mat const  & img, std::vector<Yolo::Detection> & detections);

std::vector<std::vector<float>> toVector(cv::Mat const  & img, std::vector<Yolo::Detection> & detections);


std::vector<Tn::Bbox> postProcessImgToBbox(cv::Mat const  & img, std::vector<Yolo::Detection> & detections,
                                           int classes, float nms_threshold, int net_width, int net_height);
std::vector<std::vector<float>> postProcessImgToVector(cv::Mat const  & img, std::vector<Yolo::Detection> & detections,
                                                       int classes, float nms_threshold, int net_width, int net_height);

#endif //TENSORRTYOLOV3_UTILS_HPP
