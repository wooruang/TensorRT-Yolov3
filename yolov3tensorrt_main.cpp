
#include <opencv2/opencv.hpp>

#include <Yolov3TensorRT.hpp>
#include <chrono>

int main(int argc, char* argv[])
{
    std::string engine_path = "yolov3_fp32.engine";
    std::string test_image_path = "test.jpg";

    auto input = cv::imread(test_image_path);

    Yolov3TensorRT yolo(engine_path, 608, 608, 3, 0.4, 80);
    yolo.init();

    auto start = std::chrono::system_clock::now();
    auto output = yolo.predict(input);

    for(auto const & item : output)
    {
        cv::rectangle(input,cv::Point(item.left,item.top),cv::Point(item.right,item.bot),cv::Scalar(0,0,255),2,8,0);
        std::cout << "class=" << item.classId << " prob=" << item.score*100 << std::endl;
        std::cout << "left=" << item.left << " right=" << item.right << " top=" << item.top << " bot=" << item.bot << std::endl;
    }
    auto end = std::chrono::system_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "predict total: " << milliseconds.count() << std::endl;
    cv::imshow("result", input);
    cv::waitKey(0);
}

