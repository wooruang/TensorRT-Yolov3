#include <opencv2/opencv.hpp>
#include <TrtNet.h>
#include <TensorRT/argsParser.h>
#include <TensorRT/configs.h>
#include <chrono>
#include <YoloLayer.h>
#include <TensorRT/dataReader.h>
#include <TensorRT/eval.h>
#include <TensorRT/utils.hpp>

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace Yolo;

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

int main( int argc, char* argv[] )
{
    parser::ADD_ARG_STRING("prototxt",Desc("input yolov3 deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_STRING("caffemodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("calib",Desc("calibration image List"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));
    parser::ADD_ARG_FLOAT("nms",Desc("non-maximum suppression value"),DefaultValue(to_string(NMS_THRESH)));

    //input
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_STRING("evallist",Desc("eval gt list"),DefaultValue(EVAL_LIST),ValueDesc("file"));

    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);

    string deployFile = parser::getStringValue("prototxt");
    string caffemodelFile = parser::getStringValue("caffemodel");

    vector<vector<float>> calibData;
    string calibFileList = parser::getStringValue("calib");
    string mode = parser::getStringValue("mode");
    if(calibFileList.length() > 0 && mode == "int8")
    {   
        cout << "find calibration file,loading ..." << endl;
      
        ifstream file(calibFileList);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << calibFileList << endl;
            exit(-1);
        }

        string strLine;  
        while( getline(file,strLine) )                               
        { 
            cv::Mat img = cv::imread(strLine);
            auto data = prepareImage(img, parser::getIntValue("C"), parser::getIntValue("W"), parser::getIntValue("H"));
            calibData.emplace_back(data);
        } 
        file.close();
    }

    RUN_MODE run_mode = RUN_MODE::FLOAT32;
    if(mode == "int8")
    {
        if(calibFileList.length() == 0)
            cout << "run int8 please input calibration file, will run in fp32" << endl;
        else
            run_mode = RUN_MODE::INT8;
    }
    else if(mode == "fp16")
    {
        run_mode = RUN_MODE::FLOAT16;
    }
    
    string outputNodes = parser::getStringValue("outputs");
    auto outputNames = split(outputNodes,',');
    
    //can load from file
    string saveName = "yolov3_" + mode + ".engine";

//#define LOAD_FROM_ENGINE
#ifdef LOAD_FROM_ENGINE    
    trtNet net(saveName);
#else
    trtNet net(deployFile,caffemodelFile,outputNames,calibData,run_mode);
    cout << "save Engine..." << saveName <<endl;
    net.saveEngine(saveName);
#endif

    auto start = std::chrono::system_clock::now();
    int outputCount = net.getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);

    string listFile = parser::getStringValue("evallist");
    list<string> fileNames;
    list<vector<Bbox>> groundTruth;

    if(listFile.length() > 0)
    {
        std::cout << "loading from eval list " << listFile << std::endl; 
        tie(fileNames,groundTruth) = readObjectLabelFileList(listFile);
    }
    else
    {
        string inputFileName = parser::getStringValue("input");
        fileNames.push_back(inputFileName);
    }

    list<vector<Bbox>> outputs;
    int classNum = parser::getIntValue("class");
    for (const auto& filename :fileNames)
    {
        std::cout << "process: " << filename << std::endl;

        cv::Mat img = cv::imread(filename);
        vector<float> inputData = prepareImage(img, parser::getIntValue("C"), parser::getIntValue("W"), parser::getIntValue("H"));
        if (!inputData.data())
            continue;

        net.doInference(inputData.data(), outputData.get());

        //Get Output    
        auto output = outputData.get();

        //first detect count
        int count = output[0];
        //later detect result
        vector<Detection> result;
        result.resize(count);
        memcpy(result.data(), &output[1], count*sizeof(Detection));

        auto boxes = postProcessImgToBbox(img,result,classNum, parser::getFloatValue("nms"), parser::getIntValue("W"), parser::getIntValue("H"));
        outputs.emplace_back(boxes);
    }
    auto end1 = std::chrono::system_clock::now();
    auto milliseconds1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start);
    std::cout << "aa: " << milliseconds1.count() << std::endl;

    net.printTime();



    if(groundTruth.size() > 0)
    {
        //eval map
        evalMAPResult(outputs,groundTruth,classNum,0.5f);
        evalMAPResult(outputs,groundTruth,classNum,0.75f);
    }

    if(fileNames.size() == 1)
    {
        //draw on image
        cv::Mat img = cv::imread(*fileNames.begin());
        auto bbox = *outputs.begin();
        for(const auto& item : bbox)
        {
            cv::rectangle(img,cv::Point(item.left,item.top),cv::Point(item.right,item.bot),cv::Scalar(0,0,255),3,8,0);
            cout << "class=" << item.classId << " prob=" << item.score*100 << endl;
            cout << "left=" << item.left << " right=" << item.right << " top=" << item.top << " bot=" << item.bot << endl;
        }
        auto end = std::chrono::system_clock::now();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "total: " << milliseconds.count() << std::endl;
        cv::imwrite("result.jpg",img);
        cv::imshow("result",img);
        cv::waitKey(0);
    }

    return 0;
}
