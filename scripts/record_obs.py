import numpy as np
import os
# latent projection
import numpy as np
import time

from argparse import ArgumentParser


from uniform_contact.utils.get_dino_bb import *
from uniform_contact.utils.camera import *
import plotly.graph_objects as go
import threading
import rospy
from sensor_msgs.msg import PointCloud2, Image
import tf2_ros
import geometry_msgs.msg
import copy
import pickle
import cv2
from PIL import Image as pilimage
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float64MultiArray
from franka_interface_msgs.msg import RobotState
from tf.transformations import quaternion_from_matrix
from tf.transformations import euler_from_quaternion, quaternion_matrix


# mmint_franka_calibration/scripts/record_obs.py --camera_indices 0 1

# Custom Communication class implementation
class Communication:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Create topic prefix based on camera index
        # For camera index 0, don't add any prefix
        topic_prefix = "" if camera_index == 0 else f"/camera_{camera_index}"
        
        # Create subscribers for the specified camera
        self.pointcloud_sub = rospy.Subscriber(f"{topic_prefix}/depth/color/points", PointCloud2, self.pointcloud_callback)
        self.color_image_sub = rospy.Subscriber(f"{topic_prefix}/color/image_raw", Image, self.color_image_callback)
        self.infrared1_sub = rospy.Subscriber(f"{topic_prefix}/infra1/image_rect_raw", Image, self.infrared1_callback)
        self.infrared2_sub = rospy.Subscriber(f"{topic_prefix}/infra2/image_rect_raw", Image, self.infrared2_callback)
        
        # Create locks for thread safety
        self.pointcloud_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.O_F_ext_hat_K_lock = threading.Lock()
        self.infrared1_lock = threading.Lock()
        self.infrared2_lock = threading.Lock()
        
        # Initialize data storage
        self.latest_pointcloud = None
        self.latest_color_image = None
        self.O_F_ext_hat_K = None
        self.latest_infrared1_image = None
        self.latest_infrared2_image = None
        
        # Flags to track if we've received messages
        self.received_color_image = False
        self.received_infrared1 = False
        self.received_infrared2 = False

        # subscribe to panda base to panda_hand frame using ros-tf
        self.target_frame = "panda_hand"
        self.source_frame = "panda_link0"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf_buffer)
        # Subscribe to robot state for force information
        # self.ext_force_sub = rospy.Subscriber('/robot_state_publisher_node_1/robot_state',
        #              RobotState,
        #              self.joint_position)
        
        print(f"Initialized Communication for camera {camera_index} with topic prefix '{topic_prefix}'")
    def get_wrist_pose(self):
        try:
            # Get the latest transform
            transform = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                rospy.Time(0),  # Get the latest available transform
                rospy.Duration(1.0)  # Wait up to 1 second
            )
            
            # Extract translation
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Extract rotation (quaternion)
            rotation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            # Convert quaternion to rotation matrix
            rot_matrix = quaternion_matrix(rotation)[:3, :3]
            
            # Convert quaternion to euler angles (for readability)
            euler = euler_from_quaternion(rotation)
            
            return {
                'translation': translation,
                'rotation_matrix': rot_matrix,
                'euler_angles': euler,
                'quaternion': rotation
            }
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Failed to get transform: {e}")
            return None
        
    # Callback for external force
    def O_F_ext_hat_K_callback(self, msg):
        with self.O_F_ext_hat_K_lock:
            self.O_F_ext_hat_K = np.array(msg.O_F_ext_hat_K)[:3]

    # Callback for infrared1 channel
    def infrared1_callback(self, msg):
        """Callback function to process incoming infrared1 image messages"""
        try:
            # Convert the ROS Image message to a PIL Image
            image = pilimage.frombytes('L', (msg.width, msg.height), msg.data)
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            with self.infrared1_lock:
                self.latest_infrared1_image = image_array
                self.received_infrared1 = True
        except Exception as e:
            print(f"Error processing infrared1 image: {e}")

    # Callback for infrared2 channel
    def infrared2_callback(self, msg):
        """Callback function to process incoming infrared2 image messages"""
        try:
            # Convert the ROS Image message to a PIL Image
            image = pilimage.frombytes('L', (msg.width, msg.height), msg.data)
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            with self.infrared2_lock:
                self.latest_infrared2_image = image_array
                self.received_infrared2 = True
        except Exception as e:
            print(f"Error processing infrared2 image: {e}")

    def pointcloud_callback(self, msg):
        """Callback function to process incoming PointCloud2 messages"""
        # Convert PointCloud2 to numpy array
        pc_list = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            pc_list.append(p)
        pc_array = np.array(pc_list)
        
        with self.pointcloud_lock:
            self.latest_pointcloud = pc_array

    def color_image_callback(self, msg):
        """Callback function to process incoming color image messages"""
        try:
            # Convert the ROS Image message to a PIL Image
            image = pilimage.frombytes('RGB', (msg.width, msg.height), msg.data)
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            with self.color_image_lock:
                self.latest_color_image = image_array
                self.received_color_image = True
        except Exception as e:
            print(f"Error processing color image: {e}")

    def publish_target_pose(self, target_orientation, target_position):
        """Publish target pose as TF transform"""
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "panda_link0"  # parent frame
        t.child_frame_id = "target_contact"  # child frame
        
        # Set translation
        t.transform.translation.x = target_position[0]
        t.transform.translation.y = target_position[1]
        t.transform.translation.z = target_position[2]
        
        # Convert rotation matrix to quaternion
        # Add homogeneous row to make it 4x4 matrix
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = target_orientation
        q = quaternion_from_matrix(rotation_matrix)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        # Publish transform repeatedly for 1 second
        start_time = rospy.Time.now()
        rate = rospy.Rate(30)  # 30Hz publishing rate
        while (rospy.Time.now() - start_time).to_sec() < 1.0:
            t.header.stamp = rospy.Time.now()  # Update timestamp
            self.tf_broadcaster.sendTransform(t)
            rate.sleep()
            
    def wait_for_messages(self, timeout=5.0):
        """Wait for messages to be received from the camera topics"""
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # Check at 10Hz
        
        print(f"Waiting for messages from camera {self.camera_index}...")
        
        while not rospy.is_shutdown():
            if self.received_color_image and self.received_infrared1 and self.received_infrared2:
                print(f"Received messages from all topics for camera {self.camera_index}")
                return True
                
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                print(f"Timeout waiting for messages from camera {self.camera_index}")
                return False
                
            rate.sleep()
            
        return False

# Parse command line arguments
parser = ArgumentParser(description='Record observations from specified cameras')
parser.add_argument('-c', '--camera_indices', type=int, nargs='+', default=[0], 
                    help='List of camera indices to record from (e.g., -c 0 2 for cameras 0 and 2)')
args = parser.parse_args()

# Print usage information if no arguments are provided
if len(args.camera_indices) == 0:
    parser.print_help()
    print("\nExample usage:")
    print("  python record_obs.py -c 0 2  # Record from cameras 0 and 2")
    print("  python record_obs.py --camera_indices 0 2  # Same as above")
    exit(0)

# Initialize ROS node
rospy.init_node('record_obs', anonymous=True)

# Initialize communication objects for each camera
comm_objects = {}
for camera_idx in args.camera_indices:
    comm_objects[camera_idx] = Communication(camera_index=camera_idx)

# Wait for messages from each camera
all_cameras_ready = True
for camera_idx, comm in comm_objects.items():
    if not comm.wait_for_messages(timeout=5.0):
        print(f"ERROR: No messages received from camera {camera_idx}. Exiting.")
        all_cameras_ready = False
        break

if not all_cameras_ready:
    print("Exiting due to missing camera messages.")
    exit(1)

vis = True

# CARTESIAN_IMPEDANCES = [5400.0, 5400.0, 5400.0, 50.0, 50.0, 50.0]
CARTESIAN_IMPEDANCES = [2000.0, 2000.0, 1000.0, 50.0, 50.0, 50.0]

print("Starting robot")

# Read dataset
mode = 'test'
device = 'cuda'

pca_dim = 32 #64 #16
sim_train_idx = 0
sim_keypoints_feat_all = []
apply_pca = True



if __name__ == "__main__":

    N = 10
    obs_dict = {}
    for i in range(N):
        print(f"Processing {i} / {N}")
        # with comm.pointcloud_lock:
        #     points_world = comm.latest_pointcloud
            # real_pcd_homo = np.concatenate([real_pcd, np.ones((real_pcd.shape[0], 1))], axis=1).T

        # Get the latest color image data from all cameras
        obs_dict[i] = {}
        
        for camera_idx, comm in comm_objects.items():
            with comm.color_image_lock:
                real_color_image = comm.latest_color_image

            # For camera index 0, use "camera" as the key, otherwise use "camera{idx}"
            camera_key = "camera" if camera_idx == 0 else f"camera{camera_idx}"
            
            # Use locks for infrared images as well
            with comm.infrared1_lock:
                infrared1_image = comm.latest_infrared1_image
                
            with comm.infrared2_lock:
                infrared2_image = comm.latest_infrared2_image
            
            if infrared1_image is None or infrared2_image is None:
                breakpoint()

            # get panda base to panda_hand transform
            panda_base_to_panda_hand = comm.get_wrist_pose()
            print(panda_base_to_panda_hand)
            if panda_base_to_panda_hand is None:
                breakpoint()
            
            obs_dict[i][camera_key] = {
                "infrared1": infrared1_image,
                "infrared2": infrared2_image,
                "real_color_image": real_color_image,
                "panda_base_to_panda_hand": panda_base_to_panda_hand,
            }
            
        time.sleep(0.02)

    # save obs_dict
    with open("broom_test_dict.pkl", "wb") as f:
        pickle.dump(obs_dict, f)

    