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
cd docker/script
docker build -t foundationstereo -f build_cuda12_ros1.sh .
```
3. Update Configs in `docker/config/config.yaml` and camera configs in `assets/`

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

