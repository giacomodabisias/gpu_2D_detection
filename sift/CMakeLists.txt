cmake_minimum_required(VERSION 2.8)
Project (cuda_sift CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAHS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release) 

find_package(OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${SIFT_INCLUDE_DIRS})
link_directories(${SIFT_LIB})

add_executable (cuda_sift cuda_sift.cpp sift.cpp)

target_link_libraries(cuda_sift ${OpenCV_LIBS} siftgpu GLEW GL glut)

