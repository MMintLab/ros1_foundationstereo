#!/bin/bash

if [ -z ${MMINT_FS_WORKSPACE_DIR+x} ]; then
  echo "Error: MMINT_FS_WORKSPACE_DIR environment variable is not set"
  echo "Please set MMINT_FS_WORKSPACE_DIR to your workspace directory (e.g., export MMINT_FS_WORKSPACE_DIR=/path/to/your/workspace)"
  exit 1
fi

xhost +local:root
docker run \
  -it \
  -e DISPLAY \
  -e CONSOLE \
  -e="QT_X11_NO_MITSHM=1" \
  -e ROS_HOSTNAME=192.168.1.68 \
  -e ROS_IP=192.168.1.68 \
  -e ROS_MASTER_URI=http://192.168.1.68:11311/ \
  --gpus all \
  --net host \
  --privileged \
  -v $MMINT_FS_WORKSPACE_DIR:/root/gum_ws \
  -v $MMINT_FS_WORKSPACE_DIR:/root/mmint_foundationstereo \
  -v $MMINT_FS_WORKSPACE_DIR/docker/config/terminator_config:/root/.config/terminator/config \
  -v /dev/bus/usb:/dev/bus/usb \
  $opts \
  cuda12_ros1_multi \
  bash -ci ' \
    if [ ! -z ${DISPLAY+x} ] && ( [ -z ${CONSOLE+x} ] || [ $CONSOLE == terminator ] ); then \
      ./mmint_foundationstereo/docker/scripts/start_foundationstereo_multi_terminator.sh; \
    elif [ -z ${CONSOLE+x} ] || [ $CONSOLE == tmux ]; then \
      ./frankapy-docker/scripts/start_frankapy_pc_tmux.sh; \
    fi; \
    bash \
  '
xhost -local:root
