#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_matrix

class PandaTFListener:
    def __init__(self):
        rospy.init_node('panda_tf_listener', anonymous=True)
        
        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Target frames
        self.target_frame = "panda_hand"
        self.source_frame = "panda_link0"
        
        # Rate for checking transform
        self.rate = rospy.Rate(10.0)  # 10 Hz
        
    def get_transform(self):
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

    def run(self):
        while not rospy.is_shutdown():
            transform = self.get_transform()
            
            if transform is not None:
                print("\nTransform from panda_link0 to panda_hand:")
                print(f"Translation: {transform['translation']}")
                print(f"Euler angles (RPY): {transform['euler_angles']}")
                print(f"Rotation matrix:\n{transform['rotation_matrix']}")
                print(f"Quaternion: {transform['quaternion']}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        listener = PandaTFListener()
        listener.run()
    except rospy.ROSInterruptException:
        pass 