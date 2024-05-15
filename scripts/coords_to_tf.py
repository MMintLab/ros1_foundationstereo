import numpy as np
import tf.transformations as tf

# fn = "/home/markvdm/Documents/RotatedCalibration1.9.1/build/motioncam_4_8_23_coordinates.txt"
fn = "/home/markvdm/Documents/RotatedCalibration1.9.1/build/photoneo_4_8_23_coordinates.txt"
T = np.loadtxt(fn)
# print(T)

x = T[0, 3] / 1000.0
y = T[1, 3] / 1000.0
z = T[2, 3] / 1000.0

quat = tf.quaternion_from_matrix(T)

print("%f %f %f %f %f %f %f" % (x, y, z, quat[0], quat[1], quat[2], quat[3]))
