#!/usr/bin/env python
import argparse
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
        
        # Create a static transform broadcaster
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Publish and store the transform
        self.publish_and_store_transform()

    def publish_and_store_transform(self):
        """Publishes a static transform and stores it in the TF buffer."""
        parent_frame = "panda_end_effector"
        child_frame = "apriltag_frame"

        # Create the transform message
        static_transform_stamped = geometry_msgs.msg.TransformStamped()
        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = parent_frame
        static_transform_stamped.child_frame_id = child_frame

        # Translation
        static_transform_stamped.transform.translation.x = 0
        static_transform_stamped.transform.translation.y = 0
        static_transform_stamped.transform.translation.z = 0 # 0.045

        # Rotation (Quaternion)
        static_transform_stamped.transform.rotation.x = -0.5
        static_transform_stamped.transform.rotation.y = 0.5
        static_transform_stamped.transform.rotation.z = -0.5
        static_transform_stamped.transform.rotation.w = 0.5

        # Publish the transform
        self.broadcaster.sendTransform(static_transform_stamped)
        rospy.loginfo(f"Published static transform: {parent_frame} -> {child_frame}")

        # Manually set the transform in the buffer
        self.tf_buffer.set_transform(static_transform_stamped, "default_authority")

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
    camera_frame = 'realsense_link'
    ee_frame = 'apriltag_frame'
    world_frame = 'panda_link0'

    tf_wrapper = TF2Wrapper()
    c_T_tag = tf_wrapper.get_transform_msg(camera_frame, tag_frame).transform
    w_T_tag = tf_wrapper.get_transform_msg(world_frame, ee_frame).transform
    tag_T_c = InvertTransform(c_T_tag)
    w_T_c = ComposeTransforms(w_T_tag, tag_T_c)

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
