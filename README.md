# Unitree G1 LiDAR SLAM

ROS 2 framework developed for a Final Master's Thesis focused on **LiDAR-based localization and mapping for the Unitree G1 humanoid robot**.

The system uses a **Livox MID-360 LiDAR** and its integrated IMU to evaluate different LiDAR odometry / SLAM approaches under the same experimental conditions.

The final implementation supports three front-ends:

* **Custom ICP**
* **KISS-ICP**
* **FAST-LIO2**

A common **Pose Graph back-end with loop closure and global optimization** can be enabled to evaluate the effect of trajectory optimization independently from the selected front-end.

The project is divided into two main workspaces:

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
    └── Intel RealSense
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
           ├── Custom ICP
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

The main sensor stack includes the Livox MID-360, its integrated IMU and the Intel RealSense camera.

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

`robot-slam` runs **on the external PC**.

It contains the SLAM implementations, common ROS 2 interfaces, global optimization system and benchmarking tools used in the experimental evaluation.

The main configuration is centralized in:

```text
robot-slam/config/livox_slam_config.json
```

The active SLAM front-end can be selected from this configuration without modifying the source code.

The principal custom ICP node is:

```text
robot-slam/src/g1_lidar_odometry/g1_lidar_odometry/livox_slam_pose_node.py
```

---

## Custom ICP front-end

The custom implementation was developed specifically for this project in order to have full control over the registration pipeline.

Its final architecture combines two registration stages:

```text
Current LiDAR scan
        │
        ▼
Preprocessing
        │
        ▼
IMU-assisted initialization
        │
        ▼
Scan-to-scan ICP
        │
        ▼
Incremental pose estimate
        │
        ▼
Scan-to-map refinement
        │
        ▼
Final front-end pose
```

The implementation includes point-cloud preprocessing, Livox mounting-frame correction, IMU yaw information, keyframe management and a voxelized local map.

The scan-to-scan stage provides the main incremental estimate, while scan-to-map registration can refine this prediction using the local map.

---

## KISS-ICP

KISS-ICP is integrated as an alternative LiDAR odometry front-end.

It processes the same experimental datasets as the custom ICP implementation and exposes its result through the common evaluation architecture.

This allows both methods to be compared under equivalent input conditions.

---

## FAST-LIO2

FAST-LIO2 is integrated as the LiDAR-inertial alternative.

Unlike the purely LiDAR registration approaches, FAST-LIO2 tightly combines LiDAR and inertial information to estimate the robot motion.

Its output is integrated into the same project interface used by the other front-ends, allowing FAST-LIO2, KISS-ICP and the custom ICP implementation to be evaluated using the same recorded experiments and benchmarking tools.

---

# Pose Graph optimization

The project contains a global Pose Graph back-end that can be applied after the front-end trajectory estimation.

Its main responsibilities are:

```text
Front-end trajectory
        │
        ▼
Keyframe poses
        │
        ▼
Pose Graph
        │
        ├── Sequential constraints
        │
        └── Loop-closure constraints
                    │
                    ▼
             Global optimization
                    │
                    ▼
          Optimized trajectory
```

This separation between front-end and back-end makes it possible to evaluate both:

```text
Front-end only
```

and:

```text
Front-end + Pose Graph
```

for the same dataset.

---

# ROS 2 output interface

The different front-ends are normalized into a common set of ROS 2 topics for visualization and evaluation.

The principal outputs are:

```text
/g1/slam/local_map
/g1/slam/aligned_cloud
/g1/slam/path
/g1/slam/optimized_path
```

`/g1/slam/path` represents the trajectory produced by the active front-end.

`/g1/slam/optimized_path` represents the trajectory after global Pose Graph optimization when the back-end is enabled.

The global reference frame used for visualization is:

```text
map
```

---

# Running an offline experiment

Recorded ROS 2 bags can be processed without connecting to the physical robot.

## Terminal 1 — SLAM

For the custom ICP baseline:

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam

./run_icp_slam.sh
```

The active algorithm and Pose Graph configuration can also be controlled through:

```text
config/livox_slam_config.json
```

---

## Terminal 2 — Foxglove

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project

source g1-slam/activate_g1_slam_pc_humble.sh

ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Recommended Foxglove configuration:

```text
Fixed frame: map

/g1/slam/local_map
/g1/slam/aligned_cloud
/g1/slam/path
/g1/slam/optimized_path
```

---

## Terminal 3 — ROS 2 bag

For example:

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project

source /opt/ros/humble/setup.bash

ros2 bag play bags/tfm-bag-01 --clock
```

Replace `tfm-bag-01` with the dataset to be evaluated.

---

# Recording new datasets

New data are acquired directly on the Unitree G1.

First start the sensor stack:

```bash
cd ~/robot-core
./start_g1_full_sensor_stack.sh
```

The relevant raw ROS 2 topics can then be recorded into a ROS 2 bag.

The resulting dataset can subsequently be transferred to the PC and processed repeatedly using any of the available front-ends.

This provides an important property for the experimental evaluation: every algorithm receives the same sensor sequence.

---

# Experimental datasets

The `bags/` directory contains the datasets used during the experimental evaluation.

The final experiments are organized as:

```text
bags/
├── tfm-bag-01/
├── tfm-bag-02/
├── tfm-bag-03/
├── tfm-bag-04/
├── tfm-bag-05/
├── tfm-bag-06/
└── tfm-bag-07/
```

The bags represent different trajectories and environmental conditions designed to evaluate the behaviour of the SLAM approaches under different motion and geometric scenarios.

---

# Benchmarking

The repository includes an automated benchmarking pipeline used to execute the experimental datasets and export the results.

A single bag can be evaluated using:

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam

python scripts/run_bag_benchmark.py \
    ../../bags/tfm-bag-01 \
    --config config/livox_slam_config.json
```

The complete benchmark suite can be executed with:

```bash
./scripts/run_all_benchmarks.sh
```

The benchmark system records the algorithm and optimization configuration together with the trajectory metrics in CSV format.

This makes it possible to systematically compare configurations such as:

```text
ICP
ICP + Pose Graph

KISS-ICP
KISS-ICP + Pose Graph

FAST-LIO2
FAST-LIO2 + Pose Graph
```

under the same experimental conditions.

---

# Typical evaluation workflow

The complete experimental workflow used in the project is:

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
4. Execute selected SLAM front-end
                    │
                    ▼
5. Optionally apply Pose Graph optimization
                    │
                    ▼
6. Record trajectory metrics
                    │
                    ▼
7. Repeat the experiment
                    │
                    ▼
8. Compare ICP / KISS-ICP / FAST-LIO2
```

The use of recorded datasets ensures that differences between algorithms are caused by the SLAM pipeline rather than by differences in the physical trajectory performed by the robot.

---

# Software environment

The project primarily uses:

```text
Ubuntu
ROS 2 Humble
Python
NumPy
Open3D
Livox MID-360
Unitree G1
Foxglove
KISS-ICP
FAST-LIO2
```

The robot and the processing PC use separate environments because sensor acquisition and SLAM processing have different runtime requirements.

---

# Project objective

The objective of this project is to study LiDAR-based localization and mapping for the Unitree G1 and to build a reproducible framework in which different SLAM approaches can be evaluated under equivalent conditions.

The project covers the complete pipeline from physical sensor acquisition to trajectory evaluation:

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

The final framework enables direct experimental comparison between a custom ICP implementation, KISS-ICP and FAST-LIO2, as well as analysis of the contribution of global Pose Graph optimization.
