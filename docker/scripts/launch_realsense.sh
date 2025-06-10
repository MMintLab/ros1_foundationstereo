#!/bin/bash

# Source configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../config"

# Load configuration variables
eval "$(python3 $CONFIG_DIR/export_config.py --shell)"

# Launch RealSense with configured camera serial number
roslaunch mmint_foundationstereo launch_realsense.launch camera_serial_no:=$CAMERA_SERIAL_NO