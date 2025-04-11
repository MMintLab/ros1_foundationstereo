FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive dialogs during package installs
ENV DEBIAN_FRONTEND=noninteractive


# Set the working directory to /root.
WORKDIR /root


# -------------------------------------------------------------------------
# Basic Utilities and Dependencies
# -------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    locales \
    terminator \
    tmux \
    git \
    nano \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up locales (e.g., en_US.UTF-8)
RUN locale-gen en_US en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Install Python 3 and pip.
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------------------------
# Install ROS 2 (Humble) for Ubuntu 22.04 (Jammy)
# -------------------------------------------------------------------------
# 1. Add ROS 2 apt repository
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# 2. Install ROS 2 Desktop
RUN apt-get update && \
    apt-get install -y --no-install-recommends ros-humble-desktop && \
    rm -rf /var/lib/apt/lists/*

# 3. Set up ROS 2 environment each shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# -------------------------------------------------------------------------
# Create a ROS 2 workspace (using colcon)
# -------------------------------------------------------------------------
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws

# Install colcon build tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# (Optional) Example: clone your own packages here
# RUN cd /root/ros2_ws/src && \
#     git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master && \
#     cd ~/ros2_ws && \
#     sudo apt-get install python3-rosdep -y && \
#     sudo rosdep init  && \
#     rosdep update && \
#     rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y && \
#     git checkout ros2-master && \
#     git pull && \
#     cd .. && \
#     colcon build --packages-select realsense2_camera



# Install terminator and tmux.
RUN apt-get update && \
    apt-get install -y \
        terminator \
        tmux && \
    rm -rf /var/lib/apt/lists/*
    

# Install Python 3, pip, and wget (required for Miniconda installation).
RUN apt-get update && \
    apt-get install -y python3 python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y
RUN apt install git -y && \
    apt install nano -y && \
    apt install curl -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget && \
    apt-get install -y wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

    
# Install Miniconda.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -t -i -p -y


# Install foundation stereo env. 
ARG CACHEBUST=$(date +%s)
RUN echo "Cache bust: $CACHEBUST"   
RUN cd catkin_ws/src && \
    git clone https://github.com/MMintLab/ros1_foundationstereo.git && \
    cd ros1_foundationstereo && \
    git submodule update --init --recursive && \
    cd FoundationStereo && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f environment.yml && \
    conda activate foundation_stereo && \
    python -m pip install flash-attn"
    
# Build catkin workspace.
RUN apt-get update && \
    cd ~/catkin_ws && \
    /bin/bash -c ". /opt/ros/noetic/setup.bash && \
        catkin_make" && \
    rm -rf /var/lib/apt/lists/*

# Build realsense SDK
RUN apt-get update && \
    apt-get install -y gnupg2 lsb-release software-properties-common wget
RUN sudo apt-get update && \
    apt-get install -y autoconf automake libtool pkg-config libudev-dev
    
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN apt-get install apt-transport-https

## Add the RealSense repository
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/librealsense.list

## Update and install librealsense2-dkms / librealsense2-utils
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    librealsense2-dkms \
    librealsense2-utils && \
    rm -rf /var/lib/apt/lists/*
    
# # Build librealsense2
# RUN cd ~/catkin_ws/src && \
#     git clone https://github.com/Microsoft/vcpkg.git && \
#     cd vcpkg && \
#     ./bootstrap-vcpkg.sh && \
#     ./vcpkg integrate install && \
#     ./vcpkg install realsense2

# ros realsense
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
RUN apt-get update && \
    sudo apt install ros-humble-realsense2-* -y

RUN rm -r ~/catkin_ws/src/ros1_foundationstereo 

# Optionally, set up the shell to activate the environment automatically.
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate base" >> /root/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
