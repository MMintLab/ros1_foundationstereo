#!/usr/bin/env python3

import os
import numpy as np
import torch
import cv2
import pickle
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.core.foundation_stereo import *
from omegaconf import OmegaConf
import open3d as o3d
from utils import *

# Define paths
INTRINSIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "K.txt")
EXTRINSIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "extrinsics.txt")
PCD_FRAME = 'panda_link0' #"camera_color_optical_frame"

class StereoDepthProcessor():
    def __init__(self):
        torch.autograd.set_grad_enabled(False)

        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "FoundationStereo", "pretrained_models", "model_best_bp2.pth")
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        self.model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_dir)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()

        # Initialize instance variables
        self.image_left = None
        self.image_right = None
        self.image_color = None
        self.color_intrinsic = {"fx": 381.0034484863281, "fy": 380.7884826660156, 
                                "cx": 321.30181884765625, "cy": 251.1116180419922}  
        # Store camera intrinsics
        self.depth_intrinsic = {"fx": 387.2310791015625, "fy": 387.2310791015625, 
                                "cx": 315.62371826171875, "cy": 242.35133361816406}  
        
        # camera_depth_optical_frame to camera_color_optical_frame
        extrinsics = create_transformation_matrix([0.059, 0.000, -0.000],[-0.001, -0.001, 0.000, 1.000])
        self.extrinsics = np.linalg.inv(extrinsics)

        # camera_depth_optical_frame to panda_link0
        self.extrinsics_wTc = self.get_wTc_extrinsics()
        
        # Load camera intrinsics from file
        self.load_camera_intrinsics()
    
    def get_wTc_extrinsics(self):
        with open(EXTRINSIC_PATH, 'r') as f:
            lines = f.readlines()
            extrinsics_vec = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32)
        extrinsics_wTc = create_transformation_matrix(extrinsics_vec[:3], extrinsics_vec[3:])
        return extrinsics_wTc
    
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file"""
        with open(INTRINSIC_PATH, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            self.baseline = float(lines[1])
        
        # Set depth intrinsics
        self.depth_intrinsic = {
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2]
        }
        
        # For simplicity, we'll use the same intrinsics for color camera
        # self.color_intrinsic = self.depth_intrinsic.copy()
        
        # Set extrinsics (identity matrix for now, can be updated if needed)
        # self.extrinsics = np.eye(4)

    def process_images(self, image_left, image_right, image_color=None):
        """
        Process stereo images to generate depth and point cloud
        
        Args:
            image_left: Left stereo image
            image_right: Right stereo image
            image_color: Optional color image for point cloud coloring
            
        Returns:
            depth: Depth map
            pointcloud: Point cloud (if image_color is provided)
        """
        if image_left is None or image_right is None:
            print("Error: Left or right image is None")
            return None, None
        
        scale = 1.0
        image_left = np.repeat(image_left[..., None], 3, axis=-1)
        image_right = np.repeat(image_right[..., None], 3, axis=-1)

        # Reduce image size to avoid CUDA OOM
        image_left = cv2.resize(image_left, None, fx=scale, fy=scale)
        image_right = cv2.resize(image_right, None, fx=scale, fy=scale)
        H, W = image_left.shape[:2]
        image_left_ori = image_left.copy()

        # Clear CUDA cache before processing
        torch.cuda.empty_cache()

        image_left = torch.as_tensor(image_left).cuda().float()[None].permute(0,3,1,2)
        image_right = torch.as_tensor(image_right).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(image_left.shape, divis_by=32, force_square=False)
        image_left, image_right = padder.pad(image_left, image_right)

        # Process in smaller batches if needed
        with torch.cuda.amp.autocast(True):
            disp = self.model.forward(image_left, image_right, iters=32, test_mode=True)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)

        # Scale the depth intrinsics
        depth_intrinsic_scaled = {
            "fx": self.depth_intrinsic["fx"] * scale,
            "fy": self.depth_intrinsic["fy"] * scale,
            "cx": self.depth_intrinsic["cx"],
            "cy": self.depth_intrinsic["cy"]
        }
        depth = depth_intrinsic_scaled["fx"] * self.baseline / (disp)
        
        # Align depth to color if color image is provided
        pointcloud = None
        if image_color is not None:
            aligned_depth = align_depth_to_color(depth, depth_intrinsic_scaled, self.color_intrinsic, self.extrinsics)

            H, W = aligned_depth.shape
            fx_c, fy_c, cx_c, cy_c = self.color_intrinsic['fx'], self.color_intrinsic['fy'], self.color_intrinsic['cx'], self.color_intrinsic['cy']

            # Generate depth pixel coordinates
            depth_coords = np.indices((H, W)).transpose(1, 2, 0).reshape(-1, 2)
            z = aligned_depth[depth_coords[:, 0], depth_coords[:, 1]]

            # Unproject depth pixels to 3D
            x = (depth_coords[:, 1] - cx_c) * z / fx_c
            y = (depth_coords[:, 0] - cy_c) * z / fy_c
            points_3d_transformed = np.vstack((x, y, z, np.ones_like(z)))  # (4, N)

            points_3d_valid = points_3d_transformed.T[:, :3]  # Take only x,y,z coordinates
            colors = cv2.resize(image_color, None, fx=scale, fy=scale)
            colors = colors.reshape(-1, 3) / 255.0  # Normalize color values to [0,1]
            
            # Pack RGB into a single float32
            rgb_packed = np.zeros(len(points_3d_valid), dtype=np.uint32)
            for i in range(len(points_3d_valid)):
                r, g, b = colors[i]
                rgb_packed[i] = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)

            if PCD_FRAME != "camera_color_optical_frame":
                real_pcd_homo = np.hstack((points_3d_valid, np.ones((points_3d_valid.shape[0], 1))))
                points_world_homo = (self.extrinsics_wTc @ real_pcd_homo.T).T
                points_3d_valid = points_world_homo[:, :3]

            # Create point cloud
            pointcloud = {}
            pointcloud['points'] = points_3d_valid
            pointcloud['rgb'] = rgb_packed
            
        return depth, pointcloud


def main(args=None):
    print("Stereo Depth Processor for post-processing")
    print("Initializing...")
    
    # Initialize the processor
    processor = StereoDepthProcessor()
    
    # Load dataset from pickle
    dataset_paths = ["/root/mmint_foundationstereo/data/dustpan_obs_dict.pkl",
                     "/root/mmint_foundationstereo/data/broom_obs_dict.pkl"]  # Replace with actual path
    for dataset_path in dataset_paths:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Process each item in the dataset
        for i, data in dataset.items():
            print(f"Processing item {i+1}/{len(dataset)}")
            
            # Extract images from the dataset
            # Note: Adjust these keys based on your actual dataset structure
            image_left = data["infrared1"]
            image_right = data["infrared2"]
            image_color = data["real_color_image"]
            
            if image_left is None or image_right is None:
                print(f"Skipping item {i+1}: Missing stereo images")
                continue
            
            # Process the images
            depth, pointcloud = processor.process_images(image_left, image_right, image_color)
            
            # # visualize the pointcloud
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pointcloud['points'][:, :3])
            # # Convert packed RGB uint32 to float RGB values between 0 and 1
            # rgb_float = np.zeros((len(pointcloud['rgb']), 3))
            # rgb_float[:, 0] = ((pointcloud['rgb'] >> 16) & 255) / 255.0  # Red
            # rgb_float[:, 1] = ((pointcloud['rgb'] >> 8) & 255) / 255.0   # Green 
            # rgb_float[:, 2] = (pointcloud['rgb'] & 255) / 255.0          # Blue
            # pcd.colors = o3d.utility.Vector3dVector(rgb_float)
            # o3d.visualization.draw_geometries([pcd])

            # Save or use the results as needed
            # For example:
            # np.save(f"depth_{i}.npy", depth)
            # if pointcloud is not None:
            #     np.save(f"pointcloud_{i}.npy", pointcloud)
            
            print(f"Processed item {i+1}: Depth shape {depth.shape}")
            if pointcloud is not None:
                print(f"Point cloud shape: {len(pointcloud)}")
        
            dataset[i]['pointcloud'] = pointcloud

            # if i > 10:
            #     break
        print("Processing complete!")

        # save the dataset (overwrite)
        with open(dataset_path[:-4] + '_post.pkl', 'wb') as f:
            pickle.dump(dataset, f)

if __name__ == '__main__':
    main()
