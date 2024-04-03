cmake_minimum_required(VERSION 3.18)
project(supreme LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
link_libraries(${SDL2_LIBRARIES})

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

include_directories("/home/andrew/opt/gmp-6.3.0/build/include")
link_directories("/home/andrew/opt/gmp-6.3.0/build/lib")

include_directories("/home/andrew/opt/CGBN/include")

add_executable(supreme main.cu)

