# MMint FoundationStereo (ROS1)

This package publishes ROS1 topics of foundation stereo depth estimation


## [Recommended] Docker Guide

### Setup
1. Clone and set up workspace directory environment variable
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
* `camera_serial_no` in `docker/scripts/launch_realsense.sh` 
* IPs in `docker/scripts/run_foundationstereo.sh`. In specific, these three 
```
  -e ROS_HOSTNAME=192.168.1.68 \
  -e ROS_IP=192.168.1.68 \
  -e ROS_MASTER_URI=http://192.168.1.68:11311/ \
```
* Define the topic names in `stereo.py`
* [Optional] Update extrinsics between the world (robot) and camera in `mmint_foundationstereo/assets/extrinsics.txt`. If you don't have one, update `scripts/stereo.py`'s `PCD_FRAME=camera_color_optical_frame`

### Run Foundation Stereo with ROS1

```
./run_foundationstereo.sh
```
Make sure you are not running any cameras before running! 




## Installation from Scratch
This is recommended for testing out FoundationStereo without having to deal with ROS1.

```
git clone git@github.com:MMintLab/ros1_foundationstereo.
git submodule update --init --recursive
cd FoundationStereo
pip install -e .
pip install -e FoundationStereo/

```

