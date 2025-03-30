#!/usr/bin/env python
import argparse
import numpy as np
from typing import List
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from arc_utilities.transformation_helper import *
# from mmint_camera_utils.calibration_utils.camera_calibration import CameraApriltagCalibration
# from utils import setup_panda



import rospy
import tf2_ros
import geometry_msgs.msg

class TransformHandler:
    def __init__(self):
        rospy.init_node("transform_handler", anonymous=True)
        
        # Initialize the TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        


    def get_transform(self, target_frame, source_frame):
        """Looks up the transform from source_frame to target_frame."""
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
            rospy.loginfo(f"Transform from {source_frame} to {target_frame}: {transform}")
            return transform
        except tf2_ros.LookupException as e:
            rospy.logerr(f"LookupException: {e}")
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"ExtrapolationException: {e}")



def hacky_single_detection():
    tag_frame = 'tag_0'
    # camera_frame = 'camera_link'
    camera_frame = 'camera_color_optical_frame'
    ee_frame = 'apriltag_frame'
    world_frame = 'panda_link0'

    tf_wrapper = TF2Wrapper()
    c_T_tag = tf_wrapper.get_transform_msg(camera_frame, tag_frame).transform
    w_T_tag = tf_wrapper.get_transform_msg(world_frame, ee_frame).transform
    tag_T_c = InvertTransform(c_T_tag)
    w_T_c = ComposeTransforms(w_T_tag, tag_T_c)

    # camera_pos, camera_orn = ComponentsFromTransform(w_T_c)

    tag_T_c = tf_wrapper.get_transform_msg(tag_frame, camera_frame).transform
    tag_T_w = tf_wrapper.get_transform_msg(ee_frame, world_frame).transform
    tag_T_c = InvertTransform(c_T_tag)
    c_T_w = ComposeTransforms(c_T_tag, tag_T_w)
    # w_T_c = InvertTransform(c_T_w)

    camera_pos, camera_orn = ComponentsFromTransform(w_T_c)

    print("%f %f %f %f %f %f %f" %
          (camera_pos[0], camera_pos[1], camera_pos[2], camera_orn[0], camera_orn[1], camera_orn[2], camera_orn[3]))


if __name__ == "__main__":
    handler = TransformHandler()
    
    # Wait a bit to ensure TF data is available
    rospy.sleep(1.0)

    # Try to retrieve the transform
    handler.get_transform("panda_end_effector", "apriltag_frame")
    hacky_single_detection()
    rospy.spin()



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Calibrate realsense")
#     parser.add_argument("panda_id", type=int, help="Which panda to use.")
#     parser.add_argument("camera_id", type=str, help="Camera to calibrate.")
#     parser.add_argument("cal_positions_file", type=str, help="File with calibration joint positions.")
#     args = parser.parse_args()

#     calibration_positions = mmint_utils.load_gzip_pickle(args.cal_positions_file)
#     realsense_calibration(args.panda_id, args.camera_id, calibration_positions)

    # hacky_single_detection()
    # print(record_calibration_joint_positions())
    # realsense_calibration()

    # manual_realsense_calibration()
