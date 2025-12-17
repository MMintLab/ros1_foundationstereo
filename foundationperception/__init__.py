"""
FoundationPerception - A unified package for foundation model-based perception.

This package provides interfaces for:
- FoundationStereo: Stereo depth estimation (submodule)
- FoundationPose: 6DoF object pose estimation (submodule)
- SAM3: Segment Anything Model 3 for image segmentation (submodule)
- SAM3D: 3D mesh generation from single images (submodule: sam-3d-objects)

Submodules are cloned separately. To install:
    git submodule update --init --recursive
    
Then install each submodule as needed (see README.md for details).

Example usage:
    from foundationperception.stereo import StereoDepthProcessor
    
    # For pose estimation, segmentation, and mesh generation,
    # import directly from submodules after installation:
    # from foundationpose.estimater import FoundationPose
    # from sam3.model_builder import build_sam3_image_model
    # from sam3d_objects.pipeline import InferencePipeline
"""

__version__ = "0.2.0"
__author__ = "Youngsun Wi"
__email__ = "ysunnysun56@gmail.com"

from foundationperception.stereo import StereoDepthProcessor
from foundationperception.utils import (
    create_transformation_matrix,
    quaternion_to_rotation_matrix,
    align_depth_to_color,
    denoise_depth_with_sobel,
    denoise_depth_with_sobel2,
)

__all__ = [
    "StereoDepthProcessor",
    "create_transformation_matrix",
    "quaternion_to_rotation_matrix", 
    "align_depth_to_color",
    "denoise_depth_with_sobel",
    "denoise_depth_with_sobel2",
]
