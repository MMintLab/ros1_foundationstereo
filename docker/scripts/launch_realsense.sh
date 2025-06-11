#!/bin/bash

# Source configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../config"

# Load configuration variables
eval "$(python3 $CONFIG_DIR/export_config.py --shell)"

# Use the first argument as launch file name, default to launch_realsense.launch
LAUNCH_FILE=${1:-launch_realsense.launch}

# Launch RealSense with configured camera serial number
roslaunch mmint_foundationstereo $LAUNCH_FILE camera_serial_no:=$CAMERA_SERIAL_NO