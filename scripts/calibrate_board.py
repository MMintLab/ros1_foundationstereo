#!/usr/bin/env python
import argparse
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
import arc_utilities.transformation_helper as tf_helper
from utils import setup_panda

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate the Photoneo board.")
    parser.add_argument("--panda_id", "-p", type=int, default=1, help="Panda ID.")
    args = parser.parse_args()

    rospy.init_node("calibrate_board")
    tf_wrapper = TF2Wrapper()

    robot = setup_panda(args.panda_id, has_gripper=True)

    input("Hit enter to close grasp on calibration tool...")

    # robot.gripper.homing()
    # robot.gripper.open(wait_for_result=True)
    robot.gripper.grasp(0.01, 0.01, 0.08, 0.0, wait_for_result=True)

    print("Move the robot so that the calibration tool is flush with the calibration board in the front right corner.")
    input("Press enter once grasped tool is flush with calibration board...")

    transform = tf_wrapper.get_transform(f"panda_{args.panda_id}_link0", "board_calibration_frame")
    pos, orn = tf_helper.ExtractFromMatrix(transform)

    print(f"Calibration board pose: {pos[0]:.7f} {pos[1]:.7f} {pos[2]:.7f} {orn[0]:.7f} {orn[1]:.7f} {orn[2]:.7f} {orn[3]:.7f}")
