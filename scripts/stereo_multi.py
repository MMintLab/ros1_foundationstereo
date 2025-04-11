#!/usr/bin/env python3
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
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue, Lock
import queue

# Define the topic names
ROSTOPIC_STEREO_LEFT = "/camera/infra1/image_rect_raw"
ROSTOPIC_STEREO_RIGHT = "/camera/infra2/image_rect_raw"
ROSTOPIC_FS_DEPTH = "/foundation_stereo/depth_raw"
ROSTOPIC_RS_DEPTH = "/camera/aligned_depth_to_color/image_raw"
ROSTOPIC_COLOR = "/camera/color/image_raw"
ROSTOPIC_POINTCLOUD = "/foundation_stereo/pointcloud"
BASE_ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
PCD_FRAME = 'panda_link0' #"camera_color_optical_frame"

class CameraProcessor:
    def __init__(self, camera_index, model, bridge, tf_buffer, process_queue, process_lock):
        self.camera_index = camera_index
        self.model = model
        self.bridge = bridge
        self.tf_buffer = tf_buffer
        self.process_queue = process_queue
        self.process_lock = process_lock
        
        # Define paths based on camera index
        if camera_index == 0:
            self.intrinsic_path = os.path.join(BASE_ASSETS_PATH, "K.txt")
            self.extrinsic_path = os.path.join(BASE_ASSETS_PATH, "extrinsics.txt")
        else:
            self.intrinsic_path = os.path.join(BASE_ASSETS_PATH, "multi", f"K{camera_index}.txt")
            self.extrinsic_path = os.path.join(BASE_ASSETS_PATH, "multi", f"extrinsics{camera_index}.txt")
        
        print(f"Camera {camera_index} using intrinsic path: {self.intrinsic_path}")
        print(f"Camera {camera_index} using extrinsic path: {self.extrinsic_path}")
        
        # Create topic prefix based on camera index
        topic_prefix = "" if camera_index == 0 else f"/camera_{camera_index}"
        
        # Initialize instance variables
        self.image_left = None
        self.image_right = None
        self.image_depth = None
        self.image_depth_realsense = None
        self.image_color = None
        self.color_intrinsic = None
        self.depth_intrinsic = None
        self.extrinsics = None
        
        # Create locks for thread safety
        self.image_left_lock = threading.Lock()
        self.image_right_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.camera_info_lock = threading.Lock()
        self.infra1_info_lock = threading.Lock()
        
        # Create subscribers
        self.subscription_left = rospy.Subscriber(f"{topic_prefix}/infra1/image_rect_raw", Image, self.left_callback)
        self.subscription_right = rospy.Subscriber(f"{topic_prefix}/infra2/image_rect_raw", Image, self.right_callback)
        self.subscription_rgb = rospy.Subscriber(f"{topic_prefix}/color/image_raw", Image, self.color_callback)
        self.subscription_depth = rospy.Subscriber(f"{topic_prefix}/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.subscription_camera_info = rospy.Subscriber(f"{topic_prefix}/color/camera_info", CameraInfo, self.camera_info_callback)
        self.subscription_infra1_info = rospy.Subscriber(f"{topic_prefix}/infra1/camera_info", CameraInfo, self.infra1_info_callback)
        
        # Create publishers
        self.publisher_depth = rospy.Publisher(f"{topic_prefix}/foundation_stereo/depth_raw", Image, queue_size=10)
        self.publisher_pointcloud = rospy.Publisher(f"{topic_prefix}/foundation_stereo/pointcloud", PointCloud2, queue_size=10)
        
        print(f"Initialized CameraProcessor for camera {camera_index} with topic prefix '{topic_prefix}'")

    def left_callback(self, msg):
        with self.image_left_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            self.image_left = cv_image

    def right_callback(self, msg):
        with self.image_right_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
            self.image_right = cv_image

    def color_callback(self, msg):
        with self.color_image_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            self.image_color = cv_image

    def depth_callback(self, msg):
        with self.depth_lock:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            self.image_depth_realsense = cv_image

    def camera_info_callback(self, msg):
        with self.camera_info_lock:
            K = msg.K
            self.color_intrinsic = {
                "fx": K[0],
                "fy": K[4],
                "cx": K[2],
                "cy": K[5]
            }

    def infra1_info_callback(self, msg):
        with self.infra1_info_lock:
            K = msg.K
            self.depth_intrinsic = {
                "fx": K[0],
                "fy": K[4],
                "cx": K[2],
                "cy": K[5]
            }

    def process_images(self):
        with self.image_left_lock, self.image_right_lock:
            if self.image_left is None or self.image_right is None:
                return
            
            # Put the processing task in the queue
            self.process_queue.put((self.camera_index, self.image_left, self.image_right, self.image_color))

    def process_images_worker(self):
        while True:
            try:
                # Get a task from the queue with timeout
                camera_idx, image_left, image_right, image_color = self.process_queue.get(timeout=1.0)
                
                # Acquire the process lock to ensure only one process uses the model at a time
                with self.process_lock:
                    # Process the images
                    scale = 1.0
                    image_left = np.repeat(image_left[..., None], 3, axis=-1)
                    image_right = np.repeat(image_right[..., None], 3, axis=-1)

                    # Reduce image size to avoid CUDA OOM
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

                    with torch.cuda.amp.autocast(True):
                        disp = self.model.forward(image_left, image_right, iters=32, test_mode=True)

                    disp = padder.unpad(disp.float())
                    disp = disp.data.cpu().numpy().reshape(H,W)

                    with open(self.intrinsic_path, 'r') as f:
                        lines = f.readlines()
                        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
                        baseline = float(lines[1])

                    if self.depth_intrinsic is None or self.color_intrinsic is None:
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
                        return
                    
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
                    colors = cv2.resize(image_color, None, fx=scale, fy=scale)
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
                    
                    # Clear CUDA cache after processing
                    torch.cuda.empty_cache()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process_images_worker for camera {camera_idx}: {e}")
                continue

    def get_extrinsics(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame',
                'camera_depth_optical_frame',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
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
            
            extrinsics = create_transformation_matrix(translation, quaternion)
            return extrinsics
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f"Failed to get transform for camera {self.camera_index}: {e}")
            return None

class StereoDepthNode():
    def __init__(self, camera_indices=[0]):
        rospy.sleep(3)
        torch.autograd.set_grad_enabled(False)

        # Initialize ROS node first
        rospy.init_node('stereo_depth_node', anonymous=True)

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Load the model only once
        ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "FoundationStereo", "pretrained_models", "model_best_bp2.pth")
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        self.model = FoundationStereo(cfg)
        ckpt = torch.load(ckpt_dir)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()

        # Create shared resources for parallel processing
        self.process_queue = Queue()
        self.process_lock = Lock()

        # Create CameraProcessor instances for each camera
        self.camera_processors = {}
        for camera_idx in camera_indices:
            self.camera_processors[camera_idx] = CameraProcessor(
                camera_idx, 
                self.model,
                self.bridge,
                self.tf_buffer,
                self.process_queue,
                self.process_lock
            )

        # Start worker processes
        self.num_workers = min(len(camera_indices), mp.cpu_count() - 1)  # Leave one core free
        self.workers = []
        for _ in range(self.num_workers):
            worker = Process(target=self.process_images_worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        # Create a timer to process images periodically
        rospy.Timer(rospy.Duration(0.1), self.process_all_cameras)

    def process_all_cameras(self, event):
        # Process all cameras in parallel
        for processor in self.camera_processors.values():
            processor.process_images()

    def process_images_worker(self):
        while True:
            try:
                # Get a task from the queue with timeout
                camera_idx, image_left, image_right, image_color = self.process_queue.get(timeout=1.0)
                
                # Acquire the process lock to ensure only one process uses the model at a time
                with self.process_lock:
                    # Process the images
                    scale = 1.0
                    image_left = np.repeat(image_left[..., None], 3, axis=-1)
                    image_right = np.repeat(image_right[..., None], 3, axis=-1)

                    # Reduce image size to avoid CUDA OOM
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

                    with torch.cuda.amp.autocast(True):
                        disp = self.model.forward(image_left, image_right, iters=32, test_mode=True)

                    disp = padder.unpad(disp.float())
                    disp = disp.data.cpu().numpy().reshape(H,W)

                    with open(self.intrinsic_path, 'r') as f:
                        lines = f.readlines()
                        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
                        baseline = float(lines[1])

                    if self.depth_intrinsic is None or self.color_intrinsic is None:
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
                        return
                    
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
                    colors = cv2.resize(image_color, None, fx=scale, fy=scale)
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
                    
                    # Clear CUDA cache after processing
                    torch.cuda.empty_cache()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process_images_worker: {e}")
                continue

    def __del__(self):
        # Clean up worker processes
        for worker in self.workers:
            worker.terminate()
            worker.join()

def main(args=None):
    print("ROSNode for publishing FoundationStereo depth")
    print("Initializing...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Stereo depth node for multiple cameras')
    parser.add_argument('-c', '--camera_indices', type=int, nargs='+', default=[0], 
                        help='List of camera indices to process (e.g., -c 0 2 for cameras 0 and 2)')
    args = parser.parse_args()
    
    node = StereoDepthNode(camera_indices=args.camera_indices)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down stereo depth node")
        rospy.shutdown()

if __name__ == '__main__':
    main()
