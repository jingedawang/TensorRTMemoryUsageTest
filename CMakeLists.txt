cmake_minimum_required(VERSION 3.5)
project(TensorRTMemoryUsageTest)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -std=c++11")
add_compile_options(-std=c++11)

add_executable(TensorRTMemoryUsageTest
        main.cpp)
target_link_libraries(TensorRTMemoryUsageTest
        nvinfer)
