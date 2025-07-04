#include "infer_trt10.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

InferTRT10::InferTRT10(const std::string engine_file_path, int class_num)
{
    class_num_ = class_num;

    // 读取序列化的engine文件
    std::ifstream engineFileStream(engine_file_path, std::ios::binary);
    if (!engineFileStream) 
    {
        throw std::runtime_error("Failed to open engine file");
    }

    engineFileStream.seekg(0, std::ios::end);
    size_t engineFileSize = engineFileStream.tellg();
    engineFileStream.seekg(0, std::ios::beg);

    std::vector<char> engineData(engineFileSize);
    engineFileStream.read(engineData.data(), engineFileSize);
    engineFileStream.close();

    // 解析engine文件
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engineData.data(), engineFileSize));

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

    
    auto const input_name = engine_->getIOTensorName(0);
    auto const output_name = engine_->getIOTensorName(1);
    
    auto idims = engine_->getTensorShape(input_name);
    auto odims = engine_->getTensorShape(output_name);

    cout<<idims.d[1]<<" "<<idims.d[2]<<" "<<idims.d[3]<<endl;
    cout<<odims.d[1]<<" "<<odims.d[2]<<endl;

    Dims4 inputDims = {1, idims.d[1], idims.d[2], idims.d[3]};
    Dims2 outputDims = {odims.d[2], odims.d[1]};
    context_->setInputShape(input_name, inputDims);
    
    cudaMalloc(&buffers_[0], 1 * 3 * IN_H * IN_W * sizeof(float));
    int outputSize = class_num_* OUT_CHARS;
    cudaMalloc(&buffers_[1], outputSize * sizeof(float));

    context_->setTensorAddress(input_name, buffers_[0]);
    context_->setTensorAddress(output_name, buffers_[1]);
}

InferTRT10::~InferTRT10()
{
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

bool InferTRT10::detect(cv::Mat single_image)
{
    
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int batchSize = 1;

    // Create RAII buffer manager object

    
    int outputSize = class_num_* OUT_CHARS;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while (true)
    {
        cudaEventRecord(start, 0);

        // 等待内核执行完成
        cudaDeviceSynchronize();
        cv::Mat resizedImage = single_image;
        cv::Mat save_image=single_image.clone();
        //预处理，归一化标准化，resize
        resize(resizedImage, resizedImage, Size(IN_W, IN_H), cv::INTER_CUBIC);
                
        resizedImage.convertTo(resizedImage, CV_32FC3, 1.0/255, 0);  // 归一化并转成浮点
        std::vector<float> inputData(1 * 3 * resizedImage.rows * resizedImage.cols);
        int total_size = resizedImage.rows*resizedImage.cols;
        // 遍历图像的每个像素，按行列channel的顺序拉直
        for (int y = 0; y < resizedImage.rows; y++) 
        {
            for (int x = 0; x < resizedImage.cols; x++) 
            {
                cv::Vec3f pixel = resizedImage.at<cv::Vec3f>(y, x);
                
                float b = (pixel[0]-0.5)/0.5;
                float g = (pixel[1]-0.5)/0.5;
                float r = (pixel[2]-0.5)/0.5;
                inputData[x+resizedImage.cols*y+0*total_size] = r;
                inputData[x+resizedImage.cols*y+1*total_size] = g;
                inputData[x+resizedImage.cols*y+2*total_size] = b;
            }
        }
        // 执行推理
        
        cudaMemcpyAsync(buffers_[0], inputData.data(), 1 * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream);

        context_->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        std::vector<float> outputData(1 * outputSize);
        cudaMemcpyAsync(outputData.data(), buffers_[1], 1 * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
}

long InferTRT10::get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long timestamp = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return timestamp;
}

int main(int argc, char** argv)
{
    cudaSetDevice(0);
    cv::Mat image = cv::imread("~/Downloads/20250210-111036.jpg");
    unsigned char* data;
    cudaMalloc((void**)&data, 1920 * 1080 * 1 * sizeof(unsigned char));

    InferTRT10 sample("~/Downloads/test.engine", 51);

    // sample.build();
   
    sample.detect(image);

    return 0;
}
