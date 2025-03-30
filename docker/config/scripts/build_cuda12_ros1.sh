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

# Build catkin workspace.
RUN apt-get update && \
    cd catkin_ws && \
    /bin/bash -c ". /opt/ros/noetic/setup.bash && \
        catkin_make" && \
    rm -rf /var/lib/apt/lists/*

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
    DEBIAN_FRONTEND=noninteractive apt install -y wget && \
    apt-get install -y wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

    
# Install Miniconda.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -t -i -p -y


# Install foundation stereo env.    
RUN git clone https://github.com/MMintLab/ros1_foundationstereo.git && \
    cd ros1_foundationstereo && \
    git submodule update --init --recursive && \
    cd FoundationStereo && \
    conda env create -f environment.yml && \
    conda activate foundation_stereo


# Optionally, set up the shell to activate the environment automatically.
# RUN echo "conda activate rosenv" >> ~/.bashrc
