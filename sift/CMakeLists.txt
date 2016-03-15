cmake_minimum_required(VERSION 2.8)
Project (sift_test CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release) 

set(SRC                ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE            ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${INCLUDE})

add_executable(sift_test sift_test.cpp ${SRC}/sift.cpp)

target_link_libraries(sift_test ${OpenCV_LIBS} siftgpu GLEW GL glut)

