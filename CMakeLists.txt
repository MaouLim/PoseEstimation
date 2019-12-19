cmake_minimum_required(VERSION 3.15)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "D:/OpenCV/4.1.2/build_with_contrib_mingw5.4.0/")
set(Eigen3_INCLUDE_DIRS lib/Eigen3/include/eigen3/)
set(Sophus_INCLUDE_DIRS lib/Sophus/include/)

find_package(OpenCV REQUIRED)

include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FeatureExtraction features.cpp)
target_link_libraries(FeatureExtraction ${OpenCV_LIBS})

add_executable(PoseEstimation2D2D pose_estimation_2d2d.cpp)
target_link_libraries(PoseEstimation2D2D ${OpenCV_LIBS})

add_executable(Calibration iphone_calib.cpp)
target_link_libraries(Calibration ${OpenCV_LIBS})

add_executable(PoseEstimation2D3D pose_estimation_2d3d.cpp)
target_link_libraries(PoseEstimation2D3D ${OpenCV_LIBS})