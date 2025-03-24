#!/usr/bin/env python3

import rospy
import os
# from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.signal import convolve2d
import trimesh.transformations as tra

# Define the topic names
ROSTOPIC_STEREO_LEFT = "/camera/infra1/image_rect_raw"
ROSTOPIC_STEREO_RIGHT = "/camera/infra2/image_rect_raw"
ROSTOPIC_FS_DEPTH = "/foundation_stereo_isaac_ros/depth_raw"
ROSTOPIC_RS_DEPTH = "/camera/aligned_depth_to_color/image_raw"
ROSTOPIC_COLOR = "/camera/color/image_raw"


def align_depth_to_color(depth, depth_intrinsics, color_intrinsics, extrinsics):
    """
    Aligns the depth image to the color image using camera intrinsics and extrinsics.

    Parameters:
    depth (np.ndarray): The depth image (H, W).
    color (np.ndarray): The color image (H, W, 3).
    depth_intrinsics (dict): Depth camera intrinsics (fx, fy, cx, cy).
    color_intrinsics (dict): Color camera intrinsics (fx, fy, cx, cy).
    extrinsics (np.ndarray): 4x4 transformation matrix from depth to color.

    Returns:
    np.ndarray: Aligned depth image (H, W) matching the color frame.
    """
    H, W = depth.shape
    aligned_depth = np.zeros((H, W), dtype=np.float32)

    fx_d, fy_d, cx_d, cy_d = depth_intrinsics['fx'], depth_intrinsics['fy'], depth_intrinsics['cx'], depth_intrinsics['cy']
    fx_c, fy_c, cx_c, cy_c = color_intrinsics['fx'], color_intrinsics['fy'], color_intrinsics['cx'], color_intrinsics['cy']

    # Generate depth pixel coordinates
    depth_coords = np.indices((H, W)).transpose(1, 2, 0).reshape(-1, 2)
    z = depth[depth_coords[:, 0], depth_coords[:, 1]]

    # Ignore zero depth values
    valid = z > 0
    depth_coords = depth_coords[valid]
    z = z[valid]

    # Unproject depth pixels to 3D
    x = (depth_coords[:, 1] - cx_d) * z / fx_d
    y = (depth_coords[:, 0] - cy_d) * z / fy_d
    # points_3d = np.vstack((x, y, z, np.ones_like(z)))  # (4, N)
    # points_3d_transformed = extrinsics @ points_3d
    # x_c, y_c, z_c = points_3d_transformed[0], points_3d_transformed[1], points_3d_transformed[2]


    points_3d = np.vstack((x, y, z)).T  # (N, 3)
    points_3d_transformed = tra.transform_points(points_3d, extrinsics)
    x_c, y_c, z_c = points_3d_transformed[:, 0], points_3d_transformed[:, 1], points_3d_transformed[:, 2]

    # Project into color camera
    u_c = (x_c * fx_c / z_c + cx_c).astype(int)
    v_c = (y_c * fy_c / z_c + cy_c).astype(int)

    # Filter valid projections
    valid_proj = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
    aligned_depth[v_c[valid_proj], u_c[valid_proj]] = z_c[valid_proj]

    return aligned_depth


def denoise_depth_with_sobel(depth_image: np.ndarray, depth_gradient_threshold_m_per_pixel: float = 0.5) -> np.ndarray:
    """Denoise depth with Sobel filter for derivatives."""
    new_depth_image = np.copy(depth_image)
    sobel_x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    new_depth_image[sobel_mag > depth_gradient_threshold_m_per_pixel] = 0
    return new_depth_image


def denoise_depth_with_sobel2(depth_image: np.ndarray, depth_gradient_threshold_m_per_pixel: float = 0.5) -> np.ndarray:
    """Denoise a depth image by zeroing pixels where the Sobel gradient magnitude exceeds a threshold using SciPy's convolve2d."""
    new_depth_image = np.copy(depth_image)

    # Define standard Sobel kernels for the x and y gradients
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Apply convolution using mode='same' to keep the original image dimensions,
    # and boundary='symm' for symmetric padding along the edges.
    sobel_x = convolve2d(depth_image, Kx, mode='same', boundary='symm')
    sobel_y = convolve2d(depth_image, Ky, mode='same', boundary='symm')

    # Compute the gradient magnitude
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Zero out pixels where the gradient magnitude exceeds the threshold
    new_depth_image[sobel_mag > depth_gradient_threshold_m_per_pixel] = 0
    return new_depth_image

def get_depth_foundation_stereo(image_left: np.array, image_right: np.array):

    # get current file directory
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    save_path = os.path.join(dir_path, 'result')

    left_file = os.path.join(save_path, "image_left.png")
    right_file = os.path.join(save_path, "image_right.png") 
    disparity_file = os.path.join(save_path, "image_disparity.png") 
    depth_file = os.path.join(save_path, "image_depth.png") 

    from PIL import Image
    Image.fromarray(image_left).convert('RGB').save(left_file)
    Image.fromarray(image_right).convert('RGB').save(right_file)

    disparity = np.array(disparity) # Get disparity image from FoundationStereo

    baseline = 50 / 1e3
    K = np.array(
        [634.7092895507812, 0.0, 639.3463134765625,
         0.0, 634.7092895507812, 353.44158935546875,
         0.0, 0.0, 1.0]
    ).reshape(3,3)

    scale = 1.0
    K[:2] *= scale
    depth = K[0, 0] * baseline / (disparity)
    depth = denoise_depth_with_sobel2(depth)
    return depth

class StereoDepthNode():
    def __init__(self):

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize instance variables
        self.image_left = None
        self.image_right = None
        self.image_depth = None
        self.image_depth_realsense = None
        self.image_color = None

        rospy.init_node('stereo_depth_node', anonymous=True)

        self.bridge = CvBridge()
        self.image_left = None
        self.image_right = None
        self.image_depth_realsense = None
        self.image_color = None

        self.subscription_left = rospy.Subscriber(ROSTOPIC_STEREO_LEFT, Image, self.left_callback)
        self.subscription_right = rospy.Subscriber(ROSTOPIC_STEREO_RIGHT, Image, self.right_callback)
        self.subscription_rgb = rospy.Subscriber(ROSTOPIC_COLOR, Image, self.color_callback)
        self.subscription_depth = rospy.Subscriber(ROSTOPIC_RS_DEPTH, Image, self.depth_callback)

        self.subscription_depth = rospy.Publisher(ROSTOPIC_FS_DEPTH, Image, queue_size=10)

        rospy.Timer(rospy.Duration(0.1), self.process_images)


        # # Create subscribers
        # self.subscription_left = self.create_subscription(
        #     Image,
        #     ROSTOPIC_STEREO_LEFT,
        #     self.left_callback,
        #     10)
        # self.subscription_left  # Prevent unused variable warning

        # self.subscription_right = self.create_subscription(
        #     Image,
        #     ROSTOPIC_STEREO_RIGHT,
        #     self.right_callback,
        #     10)
        # self.subscription_right  # Prevent unused variable warning

        # self.subscription_rgb = self.create_subscription(
        #     Image,
        #     ROSTOPIC_COLOR,
        #     self.color_callback,
        #     20)
        # self.subscription_rgb  # Prevent unused variable warning


        # self.subscription_depth = self.create_subscription(
        #     Image,
        #     ROSTOPIC_RS_DEPTH,
        #     self.depth_callback,
        #     20)
        # self.subscription_depth  # Prevent unused variable warning

        # # Create publisher
        # self.publisher_depth = self.create_publisher(
        #     Image,
        #     ROSTOPIC_FS_DEPTH,
        #     10)

        # Create a timer to process images periodically
        timer_period = 0.1  # seconds
        # self.timer = self.create_timer(timer_period, self.process_images)

    def left_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        self._frame_id = msg.header.frame_id
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.image_left = cv_image
        print("left infra", self.image_left.shape)

    def right_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.image_right = cv_image
        print("right infra", self.image_right.shape)


    def color_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.image_color = cv_image
        # self.camera_header = self.image_color.header
        print("color", self.image_color.shape)


    def depth_callback(self, msg):
        # Convert ROS Image message to a torch tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        # if msg.encoding == '32FC1':
        #     cv_image = 1000.0 * cv_image
        self.image_depth_realsense = cv_image
        print("[realsense] depth", self.image_depth_realsense.shape)
        # print(msg.header)


    @staticmethod
    def quaternion_to_rotation_matrix(x, y, z, w):
        """Convert a quaternion to a rotation matrix."""
        norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
        
        return R


    @staticmethod
    def create_transformation_matrix(translation, quaternion):
        """Create a 4x4 transformation matrix from a translation vector and a quaternion."""
        # Unpack translation and quaternion
        tx, ty, tz = translation
        x, y, z, w = quaternion
        
        # Compute the rotation matrix
        R = StereoDepthNode.quaternion_to_rotation_matrix(x, y, z, w)
        
        # Create the 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        
        return T

    def process_images(self):
        if self.image_left is not None and self.image_right is not None:
            # Call the placeholder function to compute depth
            self.image_depth = get_depth_foundation_stereo(self.image_left, self.image_right)

            # Convert torch tensor back to numpy array
            # depth_image_np = self.image_depth.numpy()
            depth_image_np = self.image_depth


            extrinsics =StereoDepthNode.create_transformation_matrix([1.341499, -0.063293, 0.593472],
                                                          [-0.277419 ,0.032212 ,0.959757, 0.029443])

            color_intrinsics = {"fx": 908.0057373046875, "fy": 908.169677734375, "cx": 643.6474609375, "cy": 358.378}
            depth_intrinsics = {"fx":427.5676574707031, "fy":  419.1678466796875,"cx": 427.5676574707031,"cy":  242.5962677001953}

            depth_image_np = align_depth_to_color(depth_image_np, depth_intrinsics, color_intrinsics, extrinsics)
            depth_image_np = 1000 * depth_image_np
            depth_image_np = depth_image_np.astype(np.uint16)


            # Convert numpy array to ROS Image message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image_np, encoding="16UC1")
            depth_msg.header.frame_id = "camera_1_color_optical_frame"
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            # Publish the depth image
            self.publisher_depth.publish(depth_msg)

            print("[realsense] foundation stereo depth", depth_image_np.shape)

def main(args=None):
    # rospy.init_node(args=args)

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