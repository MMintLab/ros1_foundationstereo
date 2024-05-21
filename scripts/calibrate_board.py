#!/usr/bin/env python
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from arm_robots.panda import Panda
import arc_utilities.transformation_helper as tf_helper

if __name__ == '__main__':
    rospy.init_node("calibrate_board")
    tf_wrapper = TF2Wrapper()

    input("Press enter once grasped tool is flush with calibration board...")

    transform = tf_wrapper.get_transform("panda_1_link0", "board_calibration_frame")
    pos, orn = tf_helper.ExtractFromMatrix(transform)

    print("Calibration board pose: ", pos, orn)
