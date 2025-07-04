#ifndef INFER_TRT10_H
#define INFER_TRT10_H

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include <sys/time.h>
#include <memory>

using namespace nvinfer1;

class InferTRT10
{
public:
    InferTRT10(const std::string engine_file_path, int class_num);
    ~InferTRT10();

    bool detect(cv::Mat single_image);
    long get_current_time();

public:

    std::shared_ptr<nvinfer1::IRuntime> runtime_;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> engine_; //!< The TensorRT engine used to run the network
    std::unique_ptr<nvinfer1::IExecutionContext> context_; //!< The TensorRT execution context used to execute the network
    void* buffers_[2];
    int class_num_;
    int OUT_CHARS = 101; 

    int IN_W = 160;
    int IN_H = 60;

    std::vector<std::string> plate_gt_=
    {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
        "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
        "y", "z", "SH", "JS","HA","AH","ZJ", "BJ","TJ","CQ", "HE","YN",
        "LN","HL","HN", "SD","SC","JX","HB", "GS", "SX","SN","JL","FJ",
        "GZ","GD","GX","QH","HI","NX","XZ","NM","XJ","XE","_GA","JI",
        "GN","nei","-","@"
    };

    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            delete obj;
        }
    };

};

class Logger : public nvinfer1::ILogger
{
    public:
    void log(Severity severity, const char* msg) noexcept override 
    {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;
    

#endif  // INFER_TRT10_H