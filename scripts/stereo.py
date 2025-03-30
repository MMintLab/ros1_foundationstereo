#!/usr/bin/env python3

import rospy
import os
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.core.foundation_stereo import *
from omegaconf import OmegaConf
import tf2_ros
import open3d as o3d
from utils import *

# Define the topic names
ROSTOPIC_STEREO_LEFT = "/camera/infra1/image_rect_raw"
ROSTOPIC_STEREO_RIGHT = "/camera/infra2/image_rect_raw"
ROSTOPIC_FS_DEPTH = "/foundation_stereo_isaac_ros/depth_raw"
ROSTOPIC_RS_DEPTH = "/camera/aligned_depth_to_color/image_raw"
ROSTOPIC_COLOR = "/camera/color/image_raw"
ROSTOPIC_POINTCLOUD = "/foundation_stereo_isaac_ros/pointcloud"
INTRINSIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "K.txt")
EXTRINSIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "extrinsics.txt")
PCD_FRAME = 'panda_link0' #"camera_color_optical_frame"

class StereoDepthNode():
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

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize instance variables
        self.image_left = None
        self.image_right = None
        self.image_depth = None
        self.image_depth_realsense = None
        self.image_color = None
        self.color_intrinsic = None  # Store camera intrinsics
        self.depth_intrinsic = None  # Store depth camera intrinsics
        self.extrinsics = None  # Store extrinsics
        self.extrinsics_wTc = self.get_wTc_extrinsics()
        
        rospy.init_node('stereo_depth_node', anonymous=True)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.bridge = CvBridge()
        self.image_left = None
        self.image_right = None
        self.image_depth_realsense = None
        self.image_color = None

        self.subscription_left = rospy.Subscriber(ROSTOPIC_STEREO_LEFT, Image, self.left_callback)
        self.subscription_right = rospy.Subscriber(ROSTOPIC_STEREO_RIGHT, Image, self.right_callback)
        self.subscription_rgb = rospy.Subscriber(ROSTOPIC_COLOR, Image, self.color_callback)
        self.subscription_depth = rospy.Subscriber(ROSTOPIC_RS_DEPTH, Image, self.depth_callback)
        self.subscription_camera_info = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.subscription_infra1_info = rospy.Subscriber("/camera/infra1/camera_info", CameraInfo, self.infra1_info_callback)

        self.publisher_depth = rospy.Publisher(ROSTOPIC_FS_DEPTH, Image, queue_size=10)
        self.publisher_pointcloud = rospy.Publisher(ROSTOPIC_POINTCLOUD, PointCloud2, queue_size=10)

        # Create a timer to process images periodically
        rospy.Timer(rospy.Duration(5), self.process_images)
    
    def get_wTc_extrinsics(self):
        with open(EXTRINSIC_PATH, 'r') as f:
            lines = f.readlines()
            extrinsics_vec = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32)
        extrinsics_wTc = create_transformation_matrix(extrinsics_vec[:3], extrinsics_vec[3:])
        return extrinsics_wTc

    def left_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        self._frame_id = msg.header.frame_id
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        self.image_left = cv_image

    def right_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        self.image_right = cv_image

        # print("right infra", self.image_right.shape)

    def color_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.image_color = cv_image
        # self.camera_header = self.image_color.header
        # print("color", self.image_color.shape)

    def depth_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        # if msg.encoding == '32FC1':
        #     cv_image = 1000.0 * cv_image
        self.image_depth_realsense = cv_image
        # print("[realsense] depth", self.image_depth_realsense.shape)
        # print(msg.header)

    def camera_info_callback(self, msg):
        """Callback for camera info topic to get color camera intrinsics"""
        K = msg.K
        self.color_intrinsic = {
            "fx": K[0],
            "fy": K[4],
            "cx": K[2],
            "cy": K[5]
        }

    def infra1_info_callback(self, msg):
        """Callback for infra1 camera info topic to get depth camera intrinsics"""
        K = msg.K
        self.depth_intrinsic = {
            "fx": K[0],
            "fy": K[4],
            "cx": K[2],
            "cy": K[5]
        }
        # print("depth_intrinsic", self.depth_intrinsic)

    def get_extrinsics(self):
        """Get the transformation from depth to color frame"""
        try:
            # Get the transform from depth to color frame
            transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame',
                'camera_depth_optical_frame',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            # Extract translation and rotation
            translation = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            quaternion = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            
            # Create transformation matrix
            extrinsics = create_transformation_matrix(translation, quaternion)
            return extrinsics
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get transform: {e}")
            return None

    def process_images(self, event):
        # print("image left and right",self.image_left, self.image_right)

        if self.image_left is not None and self.image_right is not None:
            
            scale = 1
            image_left = np.repeat(self.image_left[..., None], 3, axis=-1)
            image_right = np.repeat(self.image_right[..., None], 3, axis=-1)


            # Reduce image size to avoid CUDA OOM
            scale = 1.0 # Reduce image size by half
            image_left = cv2.resize(image_left, None, fx=scale, fy=scale)
            image_right = cv2.resize(image_right, None, fx=scale, fy=scale)
            H,W = image_left.shape[:2]
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


            # get the current file directory
            # with open(INTRINSIC_PATH, 'r') as f:
            with open(INTRINSIC_PATH, 'r') as f:
                lines = f.readlines()
                K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
                baseline = float(lines[1])

            if self.depth_intrinsic is None:
                rospy.logwarn("Waiting for infra1 camera info...")
                return

            if self.color_intrinsic is None:
                rospy.logwarn("Waiting for color camera info...")
                return

            # Scale the depth intrinsics
            depth_intrinsic_scaled = {
                "fx": self.depth_intrinsic["fx"] * scale,
                "fy": self.depth_intrinsic["fy"] * scale,
                "cx": self.depth_intrinsic["cx"],
                "cy": self.depth_intrinsic["cy"]
            }
            depth = depth_intrinsic_scaled["fx"] * baseline / (disp)


            # Get extrinsics from TF
            extrinsics = self.get_extrinsics()
            if extrinsics is None:
                rospy.logwarn("Failed to get extrinsics from TF")
                return
            
            # print(extrinsics, self.color_intrinsic, depth_intrinsic_scaled)
            
            aligned_depth = align_depth_to_color(depth, depth_intrinsic_scaled, self.color_intrinsic, extrinsics)

            H, W = aligned_depth.shape
            fx_c, fy_c, cx_c, cy_c = self.color_intrinsic['fx'], self.color_intrinsic['fy'], self.color_intrinsic['cx'], self.color_intrinsic['cy']

            # Generate depth pixel coordinates
            depth_coords = np.indices((H, W)).transpose(1, 2, 0).reshape(-1, 2)
            z = aligned_depth[depth_coords[:, 0], depth_coords[:, 1]]

            # Unproject depth pixels to 3D
            x = (depth_coords[:, 1] - cx_c) * z / fx_c
            y = (depth_coords[:, 0] - cy_c) * z / fy_c
            points_3d_transformed = np.vstack((x, y, z, np.ones_like(z)))  # (4, N)

            points_3d_valid = points_3d_transformed.T[:, :3] # Take only x,y,z coordinates
            colors = cv2.resize(self.image_color, None, fx=scale, fy=scale)
            colors = colors.reshape(-1, 3) / 255.0  # Normalize color values to [0,1]
            
            # Create PointCloud2 message
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = PCD_FRAME
            
            # Define the fields for the point cloud
            fields = [
                point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
                point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1)
            ]
            
            # Pack RGB into a single float32
            rgb_packed = np.zeros(len(points_3d_valid), dtype=np.uint32)
            for i in range(len(points_3d_valid)):
                r, g, b = colors[i]
                rgb_packed[i] = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            

            

            if PCD_FRAME != "camera_color_optical_frame":
                real_pcd_homo = np.hstack((points_3d_valid, np.ones((points_3d_valid.shape[0], 1))))
                points_world_homo = (self.extrinsics_wTc @ real_pcd_homo.T).T
                points_3d_valid = points_world_homo[:, :3]


            # Create point cloud message
            points_with_color = np.zeros(len(points_3d_valid), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
            points_with_color['x'] = points_3d_valid[:, 0]
            points_with_color['y'] = points_3d_valid[:, 1] 
            points_with_color['z'] = points_3d_valid[:, 2]
            points_with_color['rgb'] = rgb_packed
            pc2_msg = point_cloud2.create_cloud(header, fields, points_with_color)

            # Publish point cloud
            self.publisher_pointcloud.publish(pc2_msg)


def main(args=None):
    print("ROSNode for publishing FoundationStereo depth")
    print("Initializing...")
    node = StereoDepthNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down stereo depth node")
        rospy.shutdown()

if __name__ == '__main__':
    main()