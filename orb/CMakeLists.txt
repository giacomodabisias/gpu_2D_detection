cmake_minimum_required(VERSION 2.8)
Project (cuda_orb CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAHS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release) 

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS} )

add_executable (cuda_orb cuda_orb.cpp)

target_link_libraries(cuda_orb ${OpenCV_LIBS})
