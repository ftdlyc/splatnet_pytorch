cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(splatnet LANGUAGES CXX CUDA)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/utils/lib)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CUDA_FLAGS "-std=c++11 --gpu-architecture=sm_61")

set(INCLUDE_DIRS
        "/usr/local/cuda-8.0/include"
        "/home/lyc/virtualenv/pytorch/lib/python3.5/site-packages/torch/lib/include"
        "/home/lyc/virtualenv/pytorch/lib/python3.5/site-packages/torch/lib/include/TH"
        "/home/lyc/virtualenv/pytorch/lib/python3.5/site-packages/torch/lib/include/THC"
        "/home/lyc/virtualenv/pytorch/include/python3.5m"
        "${PROJECT_SOURCE_DIR}/utils/include")
set(LINK_LIBRARIES "")
include_directories(${INCLUDE_DIRS})
add_subdirectory(${PROJECT_SOURCE_DIR}/utils)

