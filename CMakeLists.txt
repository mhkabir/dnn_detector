cmake_minimum_required(VERSION 2.8)
project(dnn_detector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -march=native")

# search for everything we need to build the package
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  sensor_msgs
  geometry_msgs
  nodelet
  perception_msgs
  problib
)

# OpenCV
find_package(OpenCV 3.4.2 REQUIRED)

# Boost
find_package(Boost REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp cv_bridge image_transport sensor_msgs nodelet perception_msgs problib
  DEPENDS Boost Eigen3
)

###########
## Build ##
###########

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  /home/kabir/catkin_ws/devel/include
  /home/kabir/catkin_ws/src/perception/armadillo_matrix/include
  /home/kabir/catkin_ws/src/perception/problib/include
  /opt/ros/kinetic/include
)

message(${catkin_INCLUDE_DIRS})

# Node
add_executable(dnn_detector
  src/node.cpp
  src/dnn_detector.cpp
  src/utils.cpp
)

target_link_libraries(dnn_detector
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
)

# Nodelet
add_library(dnn_detector_nodelet
  src/nodelet.cpp 
  src/dnn_detector.cpp
  src/utils.cpp
)

target_link_libraries(dnn_detector_nodelet
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
)
 
#############
## Install ##
#############

install(TARGETS dnn_detector dnn_detector_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)