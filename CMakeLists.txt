cmake_minimum_required(VERSION 3.5.1)
project(points_preprocessor_usi)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
  velodyne_pointcloud
  tf2
  tf2_ros  
)

# Why is it even linking against /opt/qt/latest???

# Broken dependency chain in /opt/qt/latest/lib/Qt/lib/libQt5Test.so.5
# /opt/qt/latest/lib/Qt/lib/libQt5Test.so.5: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5: version `Qt_5.12' not found (required by /opt/qt/latest/lib/Qt/lib/libQt5Test.so.5)...
find_package(OpenCV 3 COMPONENTS core highgui)

# OpenCV seems to need qt? Doesn't help..
# find_package(Qt5 COMPONENTS Core Gui REQUIRED)

catkin_package(CATKIN_DEPENDS
	sensor_msgs
	velodyne_pointcloud
)

###########
## Build ##
###########

include_directories(
#  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

find_package(Eigen3 REQUIRED)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
add_definitions("-std=c++11")

# Enable asserts for release build
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

link_directories(${PCL_LIBRARY_DIRS})

#Ground Plane Fitter
#find_package(PCL 1.8 REQUIRED) # really need 1.8 ??
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})

add_executable(groundplanfit 
	nodes/ground_filter/groundplanfit.cpp
)

target_include_directories(groundplanfit PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(groundplanfit
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)

#Scan Line Run
add_executable(scanlinerun 
	nodes/clustering/scanlinerun.cpp
)

target_include_directories(scanlinerun PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(scanlinerun
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        # Link against not broken qt library...
        /usr/lib/x86_64-linux-gnu/libQt5Test.so
#        Qt5::Core
#        Qt5::Gui
)


