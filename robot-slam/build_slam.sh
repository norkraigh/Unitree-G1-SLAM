#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../../g1-slam/activate_g1_slam_pc_humble.sh
colcon build --symlink-install

echo "Build complete. Run: source install/setup.bash"
