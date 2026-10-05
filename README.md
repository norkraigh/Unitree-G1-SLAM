# Unitree G1 LiDAR SLAM

ROS 2 framework developed for a Final Master's Thesis focused on **LiDAR-based localization and mapping for the Unitree G1 humanoid robot**.

The system uses a **Livox MID-360 LiDAR** and its integrated IMU to evaluate different LiDAR odometry / SLAM approaches under the same experimental conditions.

The final implementation supports three front-ends:

* **LM-ICP** (Local Map ICP, custom implementation developed in this project)
* **KISS-ICP**
* **FAST-LIO2**

A common **Pose Graph back-end with loop closure and global optimization** can be enabled to evaluate the effect of trajectory optimization independently from the selected front-end.

The project is divided into two main workspaces. On the development laptop the repository lives inside `Follower_Project/robot-internal-files/`; the commands below assume that layout:

```text
Follower_Project/
├── robot-internal-files/
│   ├── robot-core/          # Sensor acquisition on the Unitree G1
│   └── robot-slam/          # SLAM, optimization and evaluation on the PC
│
├── bags/                    # Recorded experimental datasets
└── ...
```

---

## System architecture

The robot and the SLAM processing pipeline are intentionally separated.

```text
Unitree G1
    │
    ├── Livox MID-360
    │      ├── Point cloud
    │      └── IMU
    │
    └── Intel RealSense (debugging only)
           │
           ▼
      robot-core
           │
           ▼
     Raw ROS 2 topics
           │
           ├──────────────► ROS 2 bag
           │
           ▼
      External PC
           │
           ▼
      robot-slam
           │
           ├── LM-ICP (custom)
           ├── KISS-ICP
           └── FAST-LIO2
                  │
                  ▼
          Common SLAM outputs
                  │
                  ▼
          Pose Graph back-end
                  │
           Loop closure +
        global optimization
                  │
                  ▼
       Optimized trajectory
```

This separation makes it possible to process exactly the same recorded sensor data with different algorithms.

---

# `robot-core`

`robot-core` runs **on the Unitree G1**.

Its purpose is sensor acquisition and ROS 2 communication. It does not contain the SLAM algorithms evaluated in the thesis.

The main sensor stack includes the Livox MID-360, its integrated IMU and the Intel RealSense camera. The robot runs **ROS 2 Foxy**; the RealSense was integrated and validated but is not used in the final localization pipeline.

The main Livox topics are:

```text
/livox/lidar
/livox/imu
```

To start the complete sensor stack on the robot:

```bash
cd ~/robot-core
./start_g1_full_sensor_stack.sh
```

`robot-core` is only required when working with the physical robot or recording new datasets.

For offline experiments using existing bags, it does not need to be running.

---

# `robot-slam`

`robot-slam` runs **on the external PC** (ROS 2 Humble).

It contains the SLAM implementations, the common ROS 2 interface, the global optimization back-end and the benchmarking tools used in the experimental evaluation.

The main configuration is centralized in:

```text
robot-slam/config/livox_slam_config.json
```

The active front-end is selected in that file, without modifying source code:

```json
"slam": {
    "algorithm": "icp"
}
```

Valid values: `"icp"` (LM-ICP), `"kiss_icp"` or `"fast_lio2"`.

Algorithm-specific configuration files:

```text
robot-slam/config/kiss_icp_mid360.yaml
robot-slam/config/fast_lio2_mid360.yaml
```

Main source files:

```text
run_slam.py                                              # Launches the selected front-end + common back-end
src/g1_lidar_odometry/g1_lidar_odometry/livox_slam_pose_node.py   # LM-ICP node
scripts/kiss_icp_output_adapter.py                       # KISS-ICP  -> common interface
scripts/fast_lio_output_adapter.py                       # FAST-LIO2 -> common interface
scripts/global_pose_graph_backend.py                     # Loop closure + Pose Graph back-end
scripts/livox_mount_tf_publisher.py                      # Static TF for the inverted Livox mounting
scripts/run_bag_benchmark.py                             # Single-bag run + metrics
scripts/run_all_benchmarks.sh                            # Full experimental campaign
```

---

## LM-ICP front-end

LM-ICP (Local Map ICP) is the custom LiDAR odometry developed for this project. It uses point-to-point ICP (Open3D) with IMU support and explicit validation of every pose.

```text
Livox point cloud
        │
        ▼
Preprocessing
  XYZ + per-point timestamps → mounting rotation → rotational deskew (gyroscope)
  → range filter → point limit → voxel downsampling
        │
        ▼
Initial guess: previous pose + gyroscope rotation increment
        │
        ▼
Reference selection
  ├── local map not ready (start-up) → scan-to-scan ICP against previous scan
  └── local map ready               → single scan-to-map ICP against local map
        │
        ▼
Pose validation
  fitness / RMSE · motion plausible for elapsed time · yaw consistent with IMU
        │
   ┌────┴─────────────┐
 valid             rejected
   │                  │
   ▼                  ▼
Publish pose,     Pose and map unchanged; prediction advances with IMU rotation.
keyframe →        After several consecutive rejections: recovery mode
local map         (relocalization against historical/optimized map + auxiliary scan-to-scan)
```

Only one ICP is executed per scan in normal operation, which keeps processing well below the 100 ms LiDAR period.

All thresholds are defined in `livox_slam_config.json`.

---

## KISS-ICP

KISS-ICP is integrated **unmodified** as the LiDAR-only reference front-end. It receives the raw `/livox/lidar` cloud, uses its own deskewing and is configured for the MID-360 in `kiss_icp_mid360.yaml`.

The inverted Livox mounting is handled through a static TF published by `livox_mount_tf_publisher.py`. The adapter translates its native outputs into the common interface and compensates the frame in which the upstream ROS wrapper publishes its local map.

---

## FAST-LIO2

FAST-LIO2 is integrated as the tightly coupled LiDAR-inertial reference.

The official ROS 2 branch is installed with `install_fast_lio2.sh`, which records the exact commit in `FAST_LIO_VERSION.txt` and applies `scripts/patch_fast_lio2_mid360.py` so that the per-point Livox timestamps are used for motion compensation.

FAST-LIO2 works in the native sensor frame (LiDAR and IMU share the factory extrinsics), so the mounting correction is applied to its outputs by the adapter.

---

# Pose Graph back-end

A common back-end runs **at the same time** as the selected front-end and is independent of it.

```text
/g1/slam/odom + /g1/slam/aligned_cloud   (from any front-end)
        │
        ▼
Keyframes (distance / yaw criteria)
        │
        ├── Sequential odometry edges
        │
        └── Loop-closure candidates
              (nearby keyframes that are not recent)
                    │
                    ▼
              ICP verification + correction limits
                    │
                    ▼
        Global optimization (Open3D, robust to inconsistent loops)
                    │
                    ▼
        /g1/slam/optimized/path · /g1/slam/optimized/global_map
```

Because front-end and back-end run together, **every execution produces both the RAW (front-end only) and the optimized trajectory**.

---

# ROS 2 output interface

All front-ends publish the same topics:

```text
/g1/slam/pose
/g1/slam/odom
/g1/slam/path             # RAW front-end trajectory
/g1/slam/aligned_cloud
/g1/slam/local_map
/g1/slam/status
```

The back-end publishes:

```text
/g1/slam/optimized/pose
/g1/slam/optimized/odom
/g1/slam/optimized/path         # Trajectory after Pose Graph optimization
/g1/slam/optimized/global_map
/g1/slam/optimized/status
```

Global frame for visualization: `map`.

---

# Running an offline experiment

Recorded bags can be processed without the physical robot.

## Environment (every terminal)

Run this in **every** terminal, including the Foxglove bridge:

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam

source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

If a local Python environment with Open3D is used on the PC, activate it as well.

> All terminals must use the same `ROS_DOMAIN_ID`. If Foxglove connects but shows no topics, this is almost always the cause.

Select the front-end in `config/livox_slam_config.json` (`slam.algorithm`).

## Option A — 2 terminals (playback handled by the script)

**Terminal 1 — Foxglove bridge**

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

**Terminal 2 — SLAM + bag playback**

```bash
python3 scripts/run_bag_benchmark.py ../../bags/tfm-bag-00 \
    --keep-alive \
    --output-dir /tmp/foxglove_runs
```

The script starts `run_slam.py` with simulated time, waits for the nodes and replays the bag with `--clock`. `--keep-alive` keeps the nodes running after playback so the final map can be inspected. Stop with `Ctrl+C`. Use `--output-dir` to avoid writing test runs into `results/`.

## Option B — 3 terminals (manual control)

**Terminal 1 — Foxglove bridge**

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

**Terminal 2 — SLAM nodes**

```bash
python3 run_slam.py --use-sim-time
```

**Terminal 3 — Bag playback**

```bash
ros2 bag play ../../bags/tfm-bag-00 --clock --topics /livox/lidar /livox/imu
```

`--use-sim-time` and `--clock` are both required when replaying bags. Restart terminal 2 before each new run, since the map and pose graph accumulate.

## Foxglove Studio

* Connection: **Foxglove WebSocket** → `ws://localhost:8765` (or `ws://<laptop-IP>:8765` from another PC on the same network)
* Fixed frame: `map`
* Recommended topics:

```text
/g1/slam/local_map
/g1/slam/aligned_cloud
/g1/slam/path
/g1/slam/optimized/path
/g1/slam/optimized/global_map
```

---

# Recording new datasets

New data are acquired directly on the Unitree G1:

```bash
cd ~/robot-core
./start_g1_full_sensor_stack.sh
```

Record at least `/livox/lidar` and `/livox/imu` into a ROS 2 bag and transfer it to the PC. Every algorithm then receives exactly the same sensor sequence.

---

# Experimental datasets

```text
bags/
├── tfm-bag-00/   # Closed route around desks in a room, 3 passes, increasing speed
├── tfm-bag-01/   # Straight corridor with the front protector mounted
├── tfm-bag-02/   # Same corridor without protector, 180° turn, return and room entry
├── tfm-bag-03/   # Straight corridor at high speed (sport mode)
├── tfm-bag-04/   # High-speed corridor with two ~90° turns
├── tfm-bag-05/   # Two clockwise 360° in-place rotations
├── tfm-bag-06/   # Clockwise 360° rotation, slow and at maximum speed
└── tfm-bag-07/   # Counter-clockwise version of bag-06
```

---

# Benchmarking

**Single bag** (selected front-end, RAW + optimized results):

```bash
python3 scripts/run_bag_benchmark.py ../../bags/tfm-bag-01 \
    --config config/livox_slam_config.json \
    --output-dir results
```

**Full campaign** (8 bags × 3 front-ends × 10 repetitions = 240 runs):

```bash
./scripts/run_all_benchmarks.sh
```

The campaign script sets `slam.algorithm` for each run, interleaves the three front-ends inside every repetition and restores the original configuration when it finishes. Playback uses only `/livox/lidar` and `/livox/imu` at real-time rate.

Results are written to `results/final_10x/`:

* `runs/<run_id>/` — trajectories, status log, run metadata and a copy of the configuration used
* `tfm_slam_benchmark.csv` — one row per run and result mode (`raw` / `optimized`), 480 rows in total

RAW and optimized trajectories are evaluated at the same timestamps (back-end keyframes) so both can be compared directly. Metrics include trajectory length, start–end distance and start–end yaw difference.

---

# Typical evaluation workflow

```text
1. Acquire raw sensor data on the Unitree G1
                    │
                    ▼
2. Record ROS 2 bag
                    │
                    ▼
3. Replay the same bag on the PC
                    │
                    ▼
4. Run the selected front-end + common Pose Graph back-end
                    │
                    ▼
5. Record RAW and optimized trajectory metrics
                    │
                    ▼
6. Repeat the experiment
                    │
                    ▼
7. Compare LM-ICP / KISS-ICP / FAST-LIO2, with and without optimization
```

---

# Software environment

```text
Robot (Unitree G1)   ROS 2 Foxy · livox_ros_driver2 · realsense-ros
PC                   Ubuntu (22.04 for ROS 2 Humble) · ROS 2 Humble · Python · NumPy · Open3D
                     KISS-ICP · FAST-LIO2 · foxglove_bridge
```

---

# Project objective

The objective of this project is to study LiDAR-based localization and mapping for the Unitree G1 and to build a reproducible framework in which different SLAM approaches can be evaluated under equivalent conditions.

```text
Sensor acquisition
        ↓
Point-cloud processing
        ↓
LiDAR / LiDAR-inertial odometry
        ↓
Local mapping
        ↓
Loop closure
        ↓
Pose Graph optimization
        ↓
Trajectory evaluation
```

The final framework enables direct experimental comparison between LM-ICP, KISS-ICP and FAST-LIO2, as well as analysis of the contribution of global Pose Graph optimization.
