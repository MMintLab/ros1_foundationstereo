#!/usr/bin/env python3
"""
Configuration file for stereo_multi.py
Contains ROS topic names, paths, and frame settings
"""

import os

# ROS Topic Names
ROSTOPIC_STEREO_LEFT = "/camera/infra1/image_rect_raw"
ROSTOPIC_STEREO_RIGHT = "/camera/infra2/image_rect_raw"
ROSTOPIC_FS_DEPTH = "/foundation_stereo/depth_raw"
ROSTOPIC_RS_DEPTH = "/camera/aligned_depth_to_color/image_raw"
ROSTOPIC_COLOR = "/camera/color/image_raw"
ROSTOPIC_POINTCLOUD = "/foundation_stereo/pointcloud"

# Paths
BASE_ASSETS_PATH = os.path.dirname(os.path.abspath(__file__))

# Frame Settings
PCD_FRAME = 'panda_link0'  # "camera_color_optical_frame"

