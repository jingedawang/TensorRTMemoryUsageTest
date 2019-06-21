# TensorRTMemoryUsageTest
This demo shows the huge memory usage when loading a TensorRT model.

## Environment
NVIDIA Jetson TX2

## Compile
```
mkdir build
cd build
cmake ..
make
```

## Usage
```
./TensorRTMemoryUsageTest ../test.plan
```

## Results
Shell output:
```
nvidia@tegra-ubuntu:~/projects/TensorRTMemoryUsageTest/build$ ./TensorRTMemoryUsageTest ../test.plan 
getMaxBatchSize: 1
getWorkspaceSize: 0
getDeviceMemorySize: 8085760
```
Memory usage:
![](https://upload-images.jianshu.io/upload_images/1186132-556ddd3f58f8ef12.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
