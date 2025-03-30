# MMint FoundationStereo (ROS1)

This package publishes ROS1 topics of foundation stereo depth estimation


## [Recommend] Docker Guide

### Setup
1. Clone ahd Set up workspace directory environment variable
```
git clone https://github.com/MMintLab/ros1_foundationstereo.git
cd ros1_foundationstereo
echo "export MMINT_FS_WORKSPACE_DIR=$(pwd)" >> ~/.bashrc
source ~/.bashrc
```

2. Build docker image
```
cd docker
docker build -t foundationstereo -f build_cuda12_ros1.sh .
```
3. Update Configs.
* IPs in `docker/scripts/run_foundationstereo.sh`. In specific, just these three 
```
  -e ROS_HOSTNAME=192.168.1.68 \
  -e ROS_IP=192.168.1.68 \
  -e ROS_MASTER_URI=http://192.168.1.68:11311/ \
```
* `camera_serial_no` in `launch_realsense.launch` 

### Run Foundation Stereo with ROS1

```
./run_foundationstereo.sh
```
* Make sure you are not running any cameras before running! 




## Installation from Scratch
```
git submodule update --init --recursive
cd FoundationStereo

```

