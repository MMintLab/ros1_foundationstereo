cmake_minimum_required(VERSION 3.0.2)
project(foundationperception)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs
)

#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
catkin_install_python(PROGRAMS
  scripts/calibrate_realsense_single_detection.py
  scripts/record_obs.py
  scripts/stereo_multi.py
  scripts/stereo_post_multi.py
  scripts/stereo_post.py
  scripts/stereo.py
  scripts/tf_listener.py
  scripts/utils.py
  scripts/video_to_objectpose.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation
install(FILES
  launch/launch_realsense.launch
  launch/launch_realsense_calibration.launch
  launch/launch_realsense_multiple.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
