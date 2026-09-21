#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/robot-core" || {
    echo "ERROR: $HOME/robot-core was not found."
    exit 1
}

source /opt/ros/foxy/setup.bash
source install/setup.bash

if [ -f "$HOME/realsense_ws/install/setup.bash" ]; then
    source "$HOME/realsense_ws/install/setup.bash"
fi

export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "Starting G1 raw sensor stack"
echo "ROS_DISTRO=${ROS_DISTRO:-unknown}"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"

pkill -f livox_ros_driver2_node 2>/dev/null || true
pkill -f realsense2_camera_node 2>/dev/null || true
pkill -f realsense2_camera 2>/dev/null || true
sleep 2

echo "Starting Livox MID-360 (LiDAR + Livox IMU)..."
ros2 launch g1_sensors_bringup g1_lidar_stack.launch.py &
LIDAR_PID=$!

sleep 4

echo "Starting Intel RealSense..."
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_gyro:=true \
  enable_accel:=true \
  unite_imu_method:=linear_interpolation &
CAMERA_PID=$!

cleanup() {
    echo "Stopping sensor stacks..."
    kill "$LIDAR_PID" "$CAMERA_PID" 2>/dev/null || true
    wait "$LIDAR_PID" "$CAMERA_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait
