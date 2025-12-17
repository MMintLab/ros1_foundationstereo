cmake_minimum_required(VERSION 3.0.2)
project(foundationperception)

## Find catkin macros and libraries
find_package(catkin REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package()

#############
## Install ##
#############

## Mark executable scripts for installation
catkin_install_python(PROGRAMS
  scripts/video_to_objectpose.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
