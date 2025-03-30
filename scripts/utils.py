import numpy as np
import cv2
from scipy.signal import convolve2d
from FoundationStereo.core.foundation_stereo import *


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


def create_transformation_matrix(translation, quaternion):
    """Create a 4x4 transformation matrix from a translation vector and a quaternion."""
    # Unpack translation and quaternion
    tx, ty, tz = translation
    x, y, z, w = quaternion
    
    # Compute the rotation matrix
    R = quaternion_to_rotation_matrix(x, y, z, w)
    
    # Create the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T


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
    points_3d = np.vstack((x, y, z, np.ones_like(z)))  # (4, N)
    
    # Transform points using numpy matrix multiplication
    points_3d_transformed = extrinsics @ points_3d
    x_c, y_c, z_c = points_3d_transformed[0], points_3d_transformed[1], points_3d_transformed[2]

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

