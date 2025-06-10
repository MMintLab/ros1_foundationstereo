FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Set the working directory to /root.
WORKDIR /root

# Install Python 3 and pip.
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 1) install curl & gnupg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      gnupg2 \
      lsb-release && \
    rm -rf /var/lib/apt/lists/*

# 2) fetch & dearmor the ROS key
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor \
    > /usr/share/keyrings/ros-archive-keyring.gpg

# 3) add the ROS repo (signed-by our new keyring)
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros-latest.list

# 1) Force non-interactive installs
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=${DEBIAN_FRONTEND}

# 2) Preseed keyboard‐configuration answers
RUN apt-get update && \
    apt-get install -y --no-install-recommends debconf-utils && \
    echo "keyboard-configuration  keyboard-configuration/layoutcode select us" | debconf-set-selections && \
    echo "keyboard-configuration  keyboard-configuration/layout select English (US)" | debconf-set-selections && \
    echo "keyboard-configuration  keyboard-configuration/modelcode select pc105" | debconf-set-selections && \
    echo "keyboard-configuration  keyboard-configuration/variantcode select " | debconf-set-selections && \
    rm -rf /var/lib/apt/lists/*
    
# 4) now you can apt-get update and install ROS Noetic
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ros-noetic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# # Install ROS Noetic.
# RUN apt-get update && \
#     apt-get install -y \
#         curl \
#         lsb-release && \
#     sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
#     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
#     apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y ros-noetic-desktop-full && \
#     echo "source /opt/ros/noetic/setup.bash" >> $HOME/.bashrc && \
#     rm -rf /var/lib/apt/lists/*

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
RUN cd catkin_ws/src && \
    git clone https://github.com/MMintLab/ros1_foundationstereo.git && \
    cd ros1_foundationstereo && \
    git submodule update --init --recursive && \
    cd FoundationStereo && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f environment.yml && \
    conda activate foundation_stereo && \
    python -m pip install flash-attn && \
    python -m pip install -e . && \
    cd .. && \
    python -m pip install -e . "

    
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
    
# Build librealsense2
RUN cd ~/catkin_ws/src && \
    git clone https://github.com/Microsoft/vcpkg.git && \
    cd vcpkg && \
    ./bootstrap-vcpkg.sh && \
    ./vcpkg integrate install && \
    ./vcpkg install realsense2

# Install arc_utilities in the background
RUN cd && \
    git clone https://github.com/UM-ARM-Lab/arc_utilities.git && \
    cd arc_utilities && \
    pip install -e . 


# ros realsense
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
RUN apt-get update && \
    apt-get install ros-noetic-realsense2-camera -y

RUN rm -r ~/catkin_ws/src/ros1_foundationstereo 

# Optionally, set up the shell to activate the environment automatically.
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate foundation_stereo" >> ~/.bashrc
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
