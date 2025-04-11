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

# Define base paths
BASE_ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
PCD_FRAME = 'panda_link0' #"camera_color_optical_frame"

class StereoDepthProcessor():
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        torch.autograd.set_grad_enabled(False)

        # Define paths based on camera index
        if camera_index == 0:
            self.intrinsic_path = os.path.join(BASE_ASSETS_PATH, "K.txt")
            self.extrinsic_path = os.path.join(BASE_ASSETS_PATH, "extrinsics.txt")
        else:
            self.intrinsic_path = os.path.join(BASE_ASSETS_PATH, "multi", f"K{camera_index}.txt")
            self.extrinsic_path = os.path.join(BASE_ASSETS_PATH, "multi", f"extrinsics{camera_index}.txt")
        
        print(f"Camera {camera_index} using intrinsic path: {self.intrinsic_path}")
        print(f"Camera {camera_index} using extrinsic path: {self.extrinsic_path}")

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
        # Load color intrinsics from file
        color_path = os.path.join(BASE_ASSETS_PATH, "multi", f"color{self.camera_index}.txt")
        with open(color_path, 'r') as f:
            color_params = list(map(float, f.readline().strip().split()))
            self.color_intrinsic = {
                "fx": color_params[0],
                "fy": color_params[1], 
                "cx": color_params[2],
                "cy": color_params[3]
            }


        # Load depth intrinsics from file
        depth_path = os.path.join(BASE_ASSETS_PATH, "multi", f"K{self.camera_index}.txt")

        with open(depth_path, 'r') as f:
            K = np.array(list(map(float, f.readline().strip().split()))).reshape(3,3)
            self.depth_intrinsic = {
                "fx": K[0,0],
                "fy": K[1,1],
                "cx": K[0,2], 
                "cy": K[1,2]
            }


        # Load depth to color extrinsics from file
        d2c_path = os.path.join(BASE_ASSETS_PATH, "multi", f"d2c{self.camera_index}.txt") 
        with open(d2c_path, 'r') as f:
            #  rosrun tf tf_echo camera_2_depth_optical_frame camera_2_color_optical_frame
            d2c_params = list(map(float, f.readline().strip().split()))
            extrinsics = create_transformation_matrix(d2c_params[:3], d2c_params[3:])
       
        self.extrinsics = np.linalg.inv(extrinsics)

        # camera_depth_optical_frame to panda_link0
        self.extrinsics_wTc = self.get_wTc_extrinsics()
        
        # Load camera intrinsics from file
        self.load_camera_intrinsics()
    
    def get_wTc_extrinsics(self):
        try:
            with open(self.extrinsic_path, 'r') as f:
                lines = f.readlines()
                extrinsics_vec = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32)
            extrinsics_wTc = create_transformation_matrix(extrinsics_vec[:3], extrinsics_vec[3:])
            return extrinsics_wTc
        except FileNotFoundError:
            print(f"Warning: Extrinsic file not found at {self.extrinsic_path}. Using default extrinsics.")
            # Return default extrinsics if file not found
            return np.eye(4)
    
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file"""
        try:
            with open(self.intrinsic_path, 'r') as f:
                lines = f.readlines()
                self.baseline = float(lines[1])
            
        except FileNotFoundError:
            print(f"Warning: Intrinsic file not found at {self.intrinsic_path}. Using default intrinsics.")
            # Keep default intrinsics if file not found

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
    
    # Load dataset from pickle
    # dataset_paths = ["/root/mmint_foundationstereo/data/dustpan_obs_dict.pkl",
    #                  "/root/mmint_foundationstereo/data/broom_obs_dict.pkl"]  # Replace with actual path
    dataset_paths = ["/root/mmint_foundationstereo/scripts/broom_test_dict.pkl"]
    for dataset_path in dataset_paths:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # First, identify all unique camera indices in the dataset
        camera_indices = set()
        for i, data in dataset.items():
            for camera_key in data.keys():
                if camera_key.startswith("camera"):
                    if camera_key == "camera":
                        camera_indices.add(0)
                    else:
                        camera_index = int(camera_key.replace("camera", ""))
                        camera_indices.add(camera_index)
        
        print(f"Found camera indices: {sorted(list(camera_indices))}")

        # Initialize processors for each camera index once
        processors = {}
        for camera_index in camera_indices:
            print(f"Initializing processor for camera {camera_index}")
            processors[camera_index] = StereoDepthProcessor(camera_index=camera_index)
        
        # Process each item in the dataset
        for i, data in dataset.items():
            print(f"Processing item {i+1}/{len(dataset)}")
            
            # List to store all point clouds from different cameras
            all_point_clouds = []
            
            # Process each camera in the data
            for camera_key, camera_data in data.items():
                # Skip non-camera data
                if not camera_key.startswith("camera"):
                    continue
                
                # Extract camera index from key
                if camera_key == "camera":
                    camera_index = 0
                else:
                    # Extract number from "cameraX" format
                    camera_index = int(camera_key.replace("camera", ""))
                
                print(f"Processing camera {camera_index}")
                
                # Use the pre-initialized processor for this camera
                processor = processors[camera_index]
                
                # Extract images from the dataset
                image_left = camera_data["infrared1"]
                image_right = camera_data["infrared2"]
                image_color = camera_data["real_color_image"]
                wrist_pose = camera_data["panda_base_to_panda_hand"]
                
                if image_left is None or image_right is None:
                    print(f"Skipping camera {camera_index}: Missing stereo images")
                    continue
                
                # Process the images
                depth, pointcloud = processor.process_images(image_left, image_right, image_color)

                # Create point cloud object and add to the list instead of visualizing immediately
                if pointcloud is not None:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pointcloud['points'][:, :3])
                    # Convert packed RGB uint32 to float RGB values between 0 and 1
                    rgb_float = np.zeros((len(pointcloud['rgb']), 3))
                    rgb_float[:, 0] = ((pointcloud['rgb'] >> 16) & 255) / 255.0  # Red
                    rgb_float[:, 1] = ((pointcloud['rgb'] >> 8) & 255) / 255.0   # Green 
                    rgb_float[:, 2] = (pointcloud['rgb'] & 255) / 255.0          # Blue
                    pcd.colors = o3d.utility.Vector3dVector(rgb_float)
                    
                    # Add camera index as a property to distinguish point clouds
                    # pcd.points = {"camera_index": camera_index}
                    
                    # Add to the list of all point clouds
                    all_point_clouds.append(pcd)
                
                print(f"Processed camera {camera_index}: Depth shape {depth.shape}")
                if pointcloud is not None:
                    print(f"Point cloud shape: {len(pointcloud['points'])}")
                
                # Store the results in the dataset
                data[camera_key]['pointcloud'] = pointcloud
                data[camera_key]['depth'] = depth
            
            # Visualize all point clouds together
            if all_point_clouds:
                print(f"Visualizing {len(all_point_clouds)} point clouds from different cameras together")
                # o3d.visualization.draw_geometries(all_point_clouds)

                # visualize the frame wrist_pose w.r.t. panda_link0 as a frame
                wrist_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0]) 
                T_wrist_pose = np.eye(4)
                T_wrist_pose[:3, :3] = wrist_pose['rotation_matrix']
                T_wrist_pose[:3, 3] = wrist_pose['translation']
                wrist_pose_frame.transform(T_wrist_pose)
                
                # Combine wrist pose frame with point clouds into single list
                geometries = [wrist_pose_frame] + all_point_clouds
                o3d.visualization.draw_geometries(geometries)
            else:
                print("No point clouds to visualize")
        
        print("Processing complete!")

        # save the dataset (overwrite)
        with open(dataset_path[:-4] + '_post.pkl', 'wb') as f:
            pickle.dump(dataset, f)

if __name__ == '__main__':
    main()
