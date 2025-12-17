#!/usr/bin/env python3
"""
Stereo depth processor using FoundationStereo model.

This module processes stereo infrared image pairs to estimate depth maps
and optionally generate colored point clouds.
"""

import os
import numpy as np
import torch
import cv2
import pickle
from omegaconf import OmegaConf

from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.core.foundation_stereo import FoundationStereo
from foundationperception.utils import (
    create_transformation_matrix,
    denoise_depth_with_sobel2,
    align_depth_to_color,
)


class StereoDepthProcessor:
    """
    Processor for stereo depth estimation using FoundationStereo.
    
    This class handles:
    - Loading and initializing the FoundationStereo model
    - Processing stereo image pairs to generate depth maps
    - Aligning depth to color camera frame
    - Generating colored point clouds
    
    Args:
        color_intrinsic: 3x3 color camera intrinsic matrix
        depth_intrinsic: 3x3 depth camera intrinsic matrix
        extrinsics_vec: 7-element vector [tx, ty, tz, qx, qy, qz, qw] for depth-to-color transform
        baseline: Stereo baseline in meters
        extrinsics_wTc_vec: Optional world-to-camera transform vector
        model_path: Optional custom path to model checkpoint
        
    Example:
        >>> processor = StereoDepthProcessor(
        ...     color_intrinsic=K_color,
        ...     depth_intrinsic=K_depth,
        ...     extrinsics_vec=extrinsics,
        ...     baseline=0.05
        ... )
        >>> depth, pointcloud = processor.process_images(left_ir, right_ir, color_img)
    """
    
    def __init__(
        self,
        color_intrinsic: np.ndarray,
        depth_intrinsic: np.ndarray,
        extrinsics_vec: np.ndarray,
        baseline: float,
        extrinsics_wTc_vec: np.ndarray = None,
        model_path: str = None,
    ):
        torch.autograd.set_grad_enabled(False)

        # Determine model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "FoundationStereo", "pretrained_models", "model_best_bp2.pth"
            )
        
        ckpt_dir = model_path
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        self.model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_dir)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()

        # Initialize camera parameters
        self.color_intrinsic = {
            "fx": color_intrinsic[0, 0],
            "fy": color_intrinsic[1, 1],
            "cx": color_intrinsic[0, 2],
            "cy": color_intrinsic[1, 2]
        }
        
        self.depth_intrinsic = {
            "fx": depth_intrinsic[0, 0],
            "fy": depth_intrinsic[1, 1],
            "cx": depth_intrinsic[0, 2],
            "cy": depth_intrinsic[1, 2]
        }

        # Compute extrinsics (depth to color transform)
        extrinsics = create_transformation_matrix(extrinsics_vec[:3], extrinsics_vec[3:])
        self.extrinsics = np.linalg.inv(extrinsics)

        # Optional world-to-camera transform
        self.extrinsics_wTc = (
            create_transformation_matrix(extrinsics_wTc_vec[:3], extrinsics_wTc_vec[3:])
            if extrinsics_wTc_vec is not None else None
        )
        
        self.baseline = baseline

    def process_images(
        self,
        image_left: np.ndarray,
        image_right: np.ndarray,
        image_color: np.ndarray = None,
        scale: float = 1.0,
    ) -> tuple:
        """
        Process stereo images to generate depth and optionally point cloud.
        
        Args:
            image_left: Left stereo image (grayscale or RGB)
            image_right: Right stereo image (grayscale or RGB)
            image_color: Optional color image for point cloud coloring
            scale: Image scale factor (default 1.0)
            
        Returns:
            tuple: (depth_map, pointcloud_dict or None)
                - depth_map: Aligned depth map in meters
                - pointcloud_dict: Dict with 'points' and 'rgb' keys, or None
        """
        if image_left is None or image_right is None:
            print("Error: Left or right image is None")
            return None, None
        
        # Convert grayscale to 3-channel if needed
        if len(image_left.shape) == 2:
            image_left = np.repeat(image_left[..., None], 3, axis=-1)
        if len(image_right.shape) == 2:
            image_right = np.repeat(image_right[..., None], 3, axis=-1)

        # Resize if needed
        if scale != 1.0:
            image_left = cv2.resize(image_left, None, fx=scale, fy=scale)
            image_right = cv2.resize(image_right, None, fx=scale, fy=scale)
            
        H, W = image_left.shape[:2]
        
        # Clear CUDA cache before processing
        torch.cuda.empty_cache()

        # Convert to tensor and process
        image_left_tensor = torch.as_tensor(image_left).cuda().float()[None].permute(0, 3, 1, 2)
        image_right_tensor = torch.as_tensor(image_right).cuda().float()[None].permute(0, 3, 1, 2)
        
        padder = InputPadder(image_left_tensor.shape, divis_by=32, force_square=False)
        image_left_tensor, image_right_tensor = padder.pad(image_left_tensor, image_right_tensor)

        # Run inference
        with torch.cuda.amp.autocast(True):
            disp = self.model.forward(image_left_tensor, image_right_tensor, iters=32, test_mode=True)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        # Scale depth intrinsics
        depth_intrinsic_scaled = {
            "fx": self.depth_intrinsic["fx"] * scale,
            "fy": self.depth_intrinsic["fy"] * scale,
            "cx": self.depth_intrinsic["cx"],
            "cy": self.depth_intrinsic["cy"]
        }
        
        # Convert disparity to depth
        depth = depth_intrinsic_scaled["fx"] * self.baseline / disp
        depth = denoise_depth_with_sobel2(depth)
        aligned_depth = align_depth_to_color(depth, depth_intrinsic_scaled, self.color_intrinsic, self.extrinsics)

        # Generate point cloud if color image provided
        pointcloud = None
        if image_color is not None:
            pointcloud = self._generate_pointcloud(aligned_depth, image_color, scale)
            
        return aligned_depth, pointcloud

    def _generate_pointcloud(
        self,
        aligned_depth: np.ndarray,
        image_color: np.ndarray,
        scale: float = 1.0,
    ) -> dict:
        """Generate colored point cloud from depth and color images."""
        H, W = aligned_depth.shape
        fx_c = self.color_intrinsic['fx']
        fy_c = self.color_intrinsic['fy']
        cx_c = self.color_intrinsic['cx']
        cy_c = self.color_intrinsic['cy']

        # Generate pixel coordinates
        depth_coords = np.indices((H, W)).transpose(1, 2, 0).reshape(-1, 2)
        z = aligned_depth[depth_coords[:, 0], depth_coords[:, 1]]

        # Unproject to 3D
        x = (depth_coords[:, 1] - cx_c) * z / fx_c
        y = (depth_coords[:, 0] - cy_c) * z / fy_c
        points_3d = np.vstack((x, y, z, np.ones_like(z))).T[:, :3]

        # Get colors
        colors = cv2.resize(image_color, None, fx=scale, fy=scale)
        colors = colors.reshape(-1, 3) / 255.0

        # Transform to world frame if extrinsics available
        if self.extrinsics_wTc is not None:
            points_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
            points_world = (self.extrinsics_wTc @ points_homo.T).T
            points_3d = points_world[:, :3]

        return {'points': points_3d, 'rgb': colors}


def main():
    """Demo function for testing the stereo depth processor."""
    print("StereoDepthProcessor demo")
    print("Please instantiate with camera parameters and call process_images()")


if __name__ == '__main__':
    main()

