#!/usr/bin/env python
import argparse

import mmint_utils
import rospy
from arm_robots.panda import Panda


def record_calibration_joint_positions():
    rospy.init_node("record_calibration_positions")

    panda = Panda(robot_namespace='')
    panda.connect()

    calibration_joint_positions = []

    while True:
        user_in = input("Collect position: [Y]/N")
        if user_in == "" or user_in == "y" or user_in == "Y":
            joint_position = list(panda.get_state("panda_arm").joint_state.position)
            calibration_joint_positions.append(joint_position)
        else:
            break
    return calibration_joint_positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record calibration positions.")
    parser.add_argument("out_fn", type=str, help="File to save positions to.")
    args = parser.parse_args()

    cal_joint_positions = record_calibration_joint_positions()
    mmint_utils.save_gzip_pickle(cal_joint_positions, args.out_fn)
