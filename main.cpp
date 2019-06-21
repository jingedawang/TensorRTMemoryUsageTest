#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <thread>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: TensorRTUsageTest <model_path>" << std::endl;
        exit(1);
    }
    std::string model_path = argv[1];

    nvinfer1::IRuntime *runtime_;
    nvinfer1::ICudaEngine *engine_;
    Logger gLogger;

    std::ifstream reader(model_path, std::ifstream::binary);
    reader.seekg(0, reader.end);
    size_t size = reader.tellg();
    reader.seekg(0);
    char *buffer = new char[size];
    reader.read(buffer, size);
    reader.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_ != nullptr);
    engine_ = runtime_->deserializeCudaEngine(buffer, size, nullptr);

    std::cout << "getMaxBatchSize: " << engine_->getMaxBatchSize() << std::endl;
    std::cout << "getWorkspaceSize: " << engine_->getWorkspaceSize() << std::endl;
    std::cout << "getDeviceMemorySize: " << engine_->getDeviceMemorySize() << std::endl;

    while (true) {
        std::chrono::duration<int, std::milli> timespan(1000);
        std::this_thread::sleep_for(timespan);
    }

}