#!/bin/bash

if [ -z ${MMINT_FS_WORKSPACE_DIR+x} ]; then
  echo "Error: MMINT_FS_WORKSPACE_DIR environment variable is not set"
  echo "Please set MMINT_FS_WORKSPACE_DIR to your workspace directory (e.g., export MMINT_FS_WORKSPACE_DIR=/path/to/your/workspace)"
  exit 1
fi

# Source configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config"

# Load configuration variables
eval "$(python3 $CONFIG_DIR/export_config.py --shell)"

# Ensure ROS networking variables are set
if [ -z "$ROS_HOSTNAME" ]; then
  export ROS_HOSTNAME=$network_ros_hostname
fi
if [ -z "$ROS_IP" ]; then
  export ROS_IP=$network_ros_ip
fi
if [ -z "$ROS_MASTER_URI" ]; then
  export ROS_MASTER_URI=$network_ros_master_uri
fi

xhost +local:root
docker run \
  -it \
  -e DISPLAY \
  -e CONSOLE \
  -e="QT_X11_NO_MITSHM=1" \
  -e ROS_HOSTNAME=$ROS_HOSTNAME \
  -e ROS_IP=$ROS_IP \
  -e ROS_MASTER_URI=$ROS_MASTER_URI \
  --gpus all \
  --net host \
  --privileged \
  -v $MMINT_FS_WORKSPACE_DIR:/root/mmint_foundationstereo \
  -v $MMINT_FS_WORKSPACE_DIR/docker/config/terminator_config:/root/.config/terminator/config \
  -v /dev/bus/usb:/dev/bus/usb \
  $opts \
  foundationstereo \
  bash -ci ' \
    if [ ! -z ${DISPLAY+x} ] && ( [ -z ${CONSOLE+x} ] || [ $CONSOLE == terminator ] ); then \
      ./mmint_foundationstereo/docker/scripts/start_foundationstereo_terminator.sh; \
    fi; \
    bash \
  '
xhost -local:root
