# MMint FoundationStereo (ROS1)

This package publishes ROS1 topics of foundation stereo depth estimation


## Installation

### Update submodule
`git submodule update --init --recursive`



Depends on:

* [mmint_camera_utils](https://github.com/MMintLab/mmint_camera_utils)
* [mmint_franka_ros](https://github.com/MMintLab/mmint_franka_ros)
* [phoxi_camera](https://github.com/MMintLab/phoxi_camera)

## Usage

All methods require robot controllers to be running (see `mmint_franka_ros`).

### Recording Calibration Positions

When running realsense apriltag calibration or Photoneo spherical calibration, we must first record a set of calibration
positions to move the robot to.

To record calibration positions:

```bash
rosrun mmint_franka_calibration record_calibration_positions <out_fn> --panda_id <panda_id>
```

While this is running, manually move the Pandas to desired calibration positions and hit enter/yes at each location to
record that calibration position.

### Realsense Calibration

Launch the realsense and apriltag:

```bash
roslaunch mmint_franka_calibration launch_realsense.launch
```

As arguments, you can provide:

* `apriltag_package_name` - launch apriltag according to the configuration files in the given package (see
  `mmint_camera_utils` to configure)
* `apriltag_base_link` - an apriltag frame is launched relative to this provided frame (see launch files for details)
* `camera_serial_no` - serial number of the realsense camera to use

Then, run the calibration:

```bash
rosrun mmint_franka_calibration calibrate_realsense <panda_id> <camera_id> <calibration_positions_fn>
```

### Photoneo Calibration

#### Calibrating the Marker Board

*The robot should have the gripper attached! Until we print a new calibration tool.*

Place the marker board in a repeatable location. Then, launch the following to advertise the calibration frame:

```bash
roslaunch mmint_franka_calibration launch_phoxi_board_calibration.launch
```

Next, find the board calibration tool. Then launch:

```bash
rosrun mmint_franka_calibration calibrate_board.py
```

This will prompt you to grasp the calibration tool. Once the tool is grasped, manually move the robot so that the
calibration tool is flush to the front right of the marker board, so that the center of the grasp lies directly above 
the front right marker. Then, hit enter to record the position of the calibration frame.

#### Spherical Calibration

*The robot should have the spherical calibrator grasped/attached.*

Run the following to publish the sphere transform location on the robot:
```bash
roslaunch mmint_franka_calibration launch_spherical_calibration.launch
```

Then, run the calibration:
```bash
rosrun mmint_franka_calibration spherical_calibration.py <calibration_positions_fn> --panda_id <panda_id>
```
