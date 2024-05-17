import argparse

import numpy as np
import tf.transformations as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts a 4x4 matrix to a TF pose")
    parser.add_argument("filename", help="The filename of the 4x4 matrix")
    args = parser.parse_args()

    T = np.loadtxt(args.filename)

    x = T[0, 3] / 1000.0
    y = T[1, 3] / 1000.0
    z = T[2, 3] / 1000.0

    quat = tf.quaternion_from_matrix(T)

    print("%f %f %f %f %f %f %f" % (x, y, z, quat[0], quat[1], quat[2], quat[3]))
