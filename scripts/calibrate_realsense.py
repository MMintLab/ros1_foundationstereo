#!/usr/bin/env python
import argparse
from typing import List

import tqdm

import mmint_utils
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from arc_utilities.transformation_helper import *
from arm_robots.panda import Panda
from mmint_camera_utils.calibration_utils.camera_calibration import CameraApriltagCalibration


def hacky_single_detection():
    tag_frame = 'tag_0'
    camera_frame = 'camera_1_link'
    ee_frame = 'apriltag_frame'
    world_frame = 'world'

    tf_wrapper = TF2Wrapper()
    c_T_tag = tf_wrapper.get_transform_msg(camera_frame, tag_frame).transform
    w_T_tag = tf_wrapper.get_transform_msg(world_frame, ee_frame).transform
    tag_T_c = InvertTransform(c_T_tag)
    w_T_c = ComposeTransforms(w_T_tag, tag_T_c)

    camera_pos, camera_orn = ComponentsFromTransform(w_T_c)

    print("%f %f %f %f %f %f %f" %
          (camera_pos[0], camera_pos[1], camera_pos[2], camera_orn[0], camera_orn[1], camera_orn[2], camera_orn[3]))


def setup_panda():
    # Panda robot interface.
    panda = Panda(robot_namespace='')
    panda.connect()
    # panda.set_joint_impedance(DEFAULT_JOINT_IMPEDANCE)
    # panda.set_cartesian_impedance(DEFAULT_CARTESIAN_IMPEDANCE)

    return panda


def realsense_calibration(camera_id: str, calibration_joint_positions: List):
    rospy.init_node("realsense_calibration")
    panda = setup_panda()
    calibrator = CameraApriltagCalibration(tag_id=0, calibration_frame_name="apriltag_frame", parent_frame_name="world")

    panda.plan_to_joint_config("panda_arm",
                               [-0.007312947749372636, -1.3044598639153866, 0.000894327755805353, -2.6089026039848533,
                                0.0007755669311734126, 1.3083581856224271, 0.779648284295485])

    for joint_position in tqdm.tqdm(calibration_joint_positions):
        panda.plan_to_joint_config("panda_arm", joint_position)
        calibrator.take_mesurement()

        if rospy.is_shutdown():
            exit()

    calibrator.broadcast_tfs()
    print(calibrator.tfs.keys())
    transform = calibrator.tfs[camera_id]

    print("%f %f %f %f %f %f %f" %
          (transform["x"], transform["y"], transform["z"], transform["qx"], transform["qy"], transform["qz"],
           transform["qw"]))

    calibrator.finish()


def manual_realsense_calibration():
    panda = setup_panda()

    calibrator = CameraApriltagCalibration(tag_id=0, calibration_frame_name="apriltag_frame", parent_frame_name="world")

    while True:
        user_in = input("Sense: [Y]/N")
        if user_in == "" or user_in == "y" or user_in == "Y":
            calibrator.take_mesurement()
        else:
            break

    calibrator.broadcast_tfs()
    transform = calibrator.tfs['1']

    print("%f %f %f %f %f %f %f" %
          (transform["x"], transform["y"], transform["z"], transform["qx"], transform["qy"], transform["qz"],
           transform["qw"]))

    calibrator.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate realsense")
    parser.add_argument("camera_id", type=str, help="Camera to calibrate.")
    parser.add_argument("cal_positions_file", type=str, help="File with calibration joint positions.")
    args = parser.parse_args()

    calibration_positions = mmint_utils.load_gzip_pickle(args.cal_positions_file)
    realsense_calibration(args.camera_id, calibration_positions)

    # hacky_single_detection()
    # print(record_calibration_joint_positions())
    # realsense_calibration()

    # manual_realsense_calibration()
