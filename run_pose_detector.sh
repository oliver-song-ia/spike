#!/bin/bash
# Script to run pose_detector.py with proper environment setup

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Source conda setup and activate human-pose environment
source /home/wheeltec/anaconda3/etc/profile.d/conda.sh
conda activate human-pose

# Navigate to project directory
cd /home/wheeltec/Documents/spike

# Run the pose detector node
python3 pose_detector.py "$@"
