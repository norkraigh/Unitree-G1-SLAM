#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../../g1-slam/activate_g1_slam_pc_humble.sh

if [ ! -d src/FAST_LIO ]; then
  echo "Cloning official FAST-LIO ROS2 branch..."
  git clone --branch ROS2 --recursive https://github.com/hku-mars/FAST_LIO.git src/FAST_LIO
else
  echo "src/FAST_LIO already exists; keeping current checkout."
fi

git -C src/FAST_LIO rev-parse HEAD > FAST_LIO_VERSION.txt
echo "FAST-LIO upstream commit: $(cat FAST_LIO_VERSION.txt)"

python3 scripts/patch_fast_lio2_mid360.py src/FAST_LIO

# The local message-only livox_ros_driver2 package satisfies FAST-LIO's
# compile-time CustomMsg dependency. The active MID-360 path uses PointCloud2.
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo
echo "FAST-LIO2 installation complete."
echo "Run: source install/setup.bash"
echo "Check: ros2 pkg prefix fast_lio"
