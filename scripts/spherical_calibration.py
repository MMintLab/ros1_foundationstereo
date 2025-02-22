#!/usr/bin/env python
import argparse

import numpy as np
import trimesh
from tqdm import tqdm

import mmint_utils
import rospy
from mmint_tools.camera_tools.pointcloud_utils import find_best_transform
from neural_traction_fields.utils import vedo_utils
from ntf_real.sensors.phoxi_scanner import PhoxiScanner
from utils import setup_panda
from arc_utilities.tf2wrapper import TF2Wrapper
import vedo
import open3d as o3d
import tf.transformations as tr


def point_cloud_to_o3d(point_cloud, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def spherical_calibration(calibration_positions, panda_id: int, vis: bool = False):
    rospy.init_node("spherical_calibration")
    sphere_size = 0.01

    panda = setup_panda(panda_id, True)
    scanner = PhoxiScanner("phoxi_camera", "PhoXi3Dscanner_sensor", "PhoXi3Dscanner_sensor")
    tf_wrapper = TF2Wrapper()

    w_points = []
    c_points = []

    for joint_position in tqdm(calibration_positions):
        panda.plan_to_joint_config(f"panda_{panda_id}", joint_position)

        # Take a scan.
        pointcloud, _ = scanner.get_scan()

        # Get sphere location w.r.t to robot.
        w_T_s = tf_wrapper.get_transform("panda_1_link0", "sphere_frame")
        w_points.append(w_T_s[:3, 3])

        # Get approx. camera location in world frame.
        w_T_c_old = tf_wrapper.get_transform("panda_1_link0", "PhoXi3Dscanner_sensor")
        c_T_w = np.linalg.inv(w_T_c_old)

        # Compute initialization of sphere location to refine.
        c_T_s = c_T_w @ w_T_s

        sphere_mesh = trimesh.creation.icosphere(radius=sphere_size)
        # sphere_mesh.apply_transform(w_T_s)

        # Filter the point cloud.
        sphere_approx_pos = c_T_s[:3, 3]
        filter = np.linalg.norm(pointcloud - sphere_approx_pos, axis=-1) < 3 * sphere_size
        pointcloud = pointcloud[filter]
        # mmint_utils.save_gzip_pickle(pointcloud, "pointcloud.pkl.gzip")
        # pointcloud = mmint_utils.load_gzip_pickle("pointcloud.pkl.gzip")

        # Run ICP on the result.
        target_pcd = point_cloud_to_o3d(pointcloud)
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(10))
        # Adjust normals so that they point away from our current approx sphere location.
        flip = np.sum((np.asarray(target_pcd.points) - sphere_approx_pos) * np.asarray(target_pcd.normals), axis=-1) < 0
        normals = np.asarray(target_pcd.normals)
        normals[flip] *= -1
        target_pcd.normals = o3d.utility.Vector3dVector(normals)

        source_pc, source_idcs = sphere_mesh.sample(1000, return_index=True)
        source_normals = sphere_mesh.face_normals[source_idcs]
        source_pcd = point_cloud_to_o3d(source_pc, source_normals)
        # source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(10))

        # o3d.visualization.draw_geometries([target_pcd])
        # o3d.visualization.draw_geometries([source_pcd])

        res = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.001, init=c_T_s,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        # s_T_w_ = res.transformation
        # w_T_s_ = np.linalg.inv(s_T_w_)
        c_T_s_ = res.transformation
        c_points.append(c_T_s_[:3, 3])

        if vis:
            sphere_mesh_init = sphere_mesh.copy()
            sphere_mesh_init.apply_transform(c_T_s)
            sphere_mesh_final = sphere_mesh.copy()
            sphere_mesh_final.apply_transform(c_T_s_)

            sphere_mesh_vedo = vedo.Mesh([sphere_mesh_init.vertices, sphere_mesh_init.faces], c="red", alpha=0.5)
            sphere_mesh_final_vedo = vedo.Mesh([sphere_mesh_final.vertices, sphere_mesh_final.faces], c="green",
                                               alpha=0.5)

            # Visualize the point cloud.
            vedo_plt = vedo.Plotter()
            vedo_plt.at(0).add(
                vedo.Points(pointcloud, c="black"),  # vedo_utils.draw_origin(0.1),
                vedo_utils.draw_pose(matrix=c_T_s, scale=0.06),
                sphere_mesh_vedo,
                sphere_mesh_final_vedo,
            )
            vedo_plt.camera.SetFocalPoint(*sphere_approx_pos)
            vedo_plt.camera.SetPosition(sphere_approx_pos[0] - 0.1, sphere_approx_pos[1], sphere_approx_pos[2])
            vedo_plt.interactive().close()

    # TODO: Address if less than 3 points are collected.
    t, R = find_best_transform(np.array(w_points), np.array(c_points))
    R_ext = np.eye(4)
    R_ext[:3, :3] = R
    quat = tr.quaternion_from_matrix(R_ext)
    print(f"Transform: {t[0]:.4f} {t[1]:.4f} {t[2]:.4f} {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cal_positions_file", type=str, help="File with calibration joint positions.")
    parser.add_argument("--panda_id", type=int, default=1, help="Panda ID")
    parser.add_argument("--vis", "-v", action="store_true", help="Visualize the point cloud.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    calibration_positions_ = mmint_utils.load_gzip_pickle(args.cal_positions_file)
    spherical_calibration(calibration_positions_, args.panda_id, args.vis)
