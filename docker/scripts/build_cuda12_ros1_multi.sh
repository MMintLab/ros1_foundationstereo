FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Set the working directory to /root.
WORKDIR /root

# Install Python 3 and pip.
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*


# Install ROS Noetic.
RUN apt-get update && \
    apt-get install -y \
        curl \
        lsb-release && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ros-noetic-desktop-full && \
    echo "source /opt/ros/noetic/setup.bash" >> $HOME/.bashrc && \
    rm -rf /var/lib/apt/lists/*

# Create catkin workspace.
RUN mkdir -p catkin_ws/src



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

# Build catkin workspace.
RUN apt-get update && \
    cd ~/catkin_ws && \
    /bin/bash -c ". /opt/ros/noetic/setup.bash && \
        catkin_make" && \
    rm -rf /var/lib/apt/lists/*

# Build realsense SDK - install dependencies
RUN apt-get update && \
    apt-get install -y \
        gnupg2 \
        lsb-release \
        software-properties-common \
        wget \
        autoconf \
        automake \
        libtool \
        pkg-config \
        libudev-dev \
        apt-transport-https \
        ca-certificates \
        cmake \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

## Build librealsense2 from source (more reliable than apt repo)
RUN cd /tmp && \
    git clone https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    git checkout v2.54.2 && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/librealsense

# ros realsense
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
RUN apt-get update && \
    apt-get install ros-noetic-realsense2-camera -y

RUN cd catkin_ws/src && \
    git clone -b dev https://github.com/MMintLab/ros1_foundationstereo.git && \
    cd ros1_foundationstereo && \
    git submodule update --init --recursive && \
    cd FoundationStereo && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml"

# Build flash-attn from source to avoid GLIBC compatibility issues
# Uninstall first in case it was installed via environment.yml, then build from source
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda run -n foundation_stereo python -m pip uninstall -y flash-attn || true && \
    MAX_JOBS=4 conda run -n foundation_stereo python -m pip install flash-attn --no-build-isolation"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda run -n foundation_stereo python -m pip install rospkg"

# Clear torch hub cache to force fresh download of dinov2
RUN rm -rf /root/.cache/torch/hub/facebookresearch_dinov2_main

RUN cd ~/catkin_ws/src/ros1_foundationstereo && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda run -n foundation_stereo pip install -e . &&\
    cd ~/catkin_ws/src/ros1_foundationstereo/FoundationStereo && \
    conda run -n foundation_stereo python -m pip install -e ."
    
   
# Optionally, set up the shell to activate the environment automatically.
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate foundation_stereo" >> ~/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc 

