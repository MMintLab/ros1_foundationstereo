#!/bin/bash

if [ -z ${FRANKAPY_DIR+x} ]; then
  opts=""
else
  opts="-v $FRANKAPY_DIR:/root/frankapy"
fi


FRANKAPY_DOCKER_DIR=$( cd $( dirname ${BASH_SOURCE[0]} )/.. && pwd )
xhost +local:root
docker run \
  -it \
  -e DISPLAY \
  -e CONSOLE \
  -e ROS_HOSTNAME=192.168.1.68 \
  -e ROS_IP=192.168.1.68 \
  -e ROS_MASTER_URI=http://192.168.1.68:11311/ \
  --gpus all \
  --net host \
  --privileged \
  -v /run/user/1000/at-spi:/run/user/1000/at-spi \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /home/young/gum_ws:/root/gum_ws \
  -v $FRANKAPY_DOCKER_DIR:/root/frankapy-docker \
  -v $FRANKAPY_DOCKER_DIR/config/terminator_config:/root/.config/terminator/config \
  $opts \
  frankapy_stereo \
  bash -ci ' \
    if [ ! -z ${DISPLAY+x} ] && ( [ -z ${CONSOLE+x} ] || [ $CONSOLE == terminator ] ); then \
      ./frankapy-docker/scripts/start_frankapy_pc_terminator.sh; \
    elif [ -z ${CONSOLE+x} ] || [ $CONSOLE == tmux ]; then \
      ./frankapy-docker/scripts/start_frankapy_pc_tmux.sh; \
    fi; \
    bash \
  '
xhost -local:root
