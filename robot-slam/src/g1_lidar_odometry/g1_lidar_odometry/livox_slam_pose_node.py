import json
import math
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker


class LivoxSlamPoseNode(Node):
    """Custom ICP front-end for the TFM.

    Processing policy:
      - bootstrap with scan-to-scan ICP;
      - once enough keyframes exist, use the rolling local map as the
        PRIMARY ICP target;
      - once the rolling map is available, try scan-to-map first and run
        scan-to-scan only as a fallback after a rejected primary registration;
        this avoids paying for two ICP registrations on every healthy scan;
      - keep validated/public pose and auxiliary tracking guess separate;
      - update the scan-to-scan reference even when scan-to-map is rejected,
        so tracking can continue through short losses during turns;
      - use the three IMU angular-rate components only as a rotational
        initial guess / short-loss fallback; ICP estimates the full 6-DoF pose;
      - the local map contains ONLY the most recent N keyframes;
      - keyframes are selected from XY displacement + yaw only, while ICP
        registration itself remains full 6-DoF;
      - accepted poses only are published and inserted into path/local map.
    """

    def __init__(self):
        super().__init__("livox_slam_pose_node")

        # ============================================================
        # CONFIGURATION
        # ============================================================

        self.declare_parameter("config_file", "")
        config_file = self.get_parameter("config_file").value
        self.config = self.load_json_config(str(config_file))

        self.slam_algorithm = str(
            self.cfg("slam.algorithm", "icp")
        ).strip().lower()

        if self.slam_algorithm != "icp":
            raise RuntimeError(
                "livox_slam_pose_node.py implements only the custom ICP "
                f"front-end, but slam.algorithm='{self.slam_algorithm}'."
            )

        self.reset_config = self.config.get("reset", {})
        self.enable_auto_reset_on_clock_jump = bool(
            self.reset_config.get("enable_auto_reset_on_clock_jump", True)
        )
        self.clock_jump_threshold_sec = float(
            self.reset_config.get("clock_jump_threshold_sec", 1.0)
        )
        self.clear_local_map_on_reset = bool(
            self.reset_config.get("clear_local_map", True)
        )
        self.clear_path_on_reset = bool(
            self.reset_config.get("clear_path", True)
        )
        self.reset_pose_to_identity = bool(
            self.reset_config.get("reset_pose_to_identity", True)
        )

        self.declare_runtime_parameters()

        # Input / frames
        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.corrected_lidar_frame = str(
            self.get_parameter("corrected_lidar_frame").value
        )

        # ICP
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.icp_max_correspondence_distance = float(
            self.get_parameter("icp_max_correspondence_distance").value
        )
        self.icp_max_iterations = int(
            self.get_parameter("icp_max_iterations").value
        )
        self.min_fitness = float(self.get_parameter("min_fitness").value)
        self.cloud_skip = int(self.get_parameter("cloud_skip").value)
        # Backward compatible: the old JSON key was
        # ``use_imu_yaw_initial_guess``. In 6-DoF mode the IMU initial guess
        # uses all three gyroscope axes (roll/pitch/yaw).
        self.use_imu_rotation_initial_guess = bool(
            self.get_parameter("use_imu_rotation_initial_guess").value
        )
        self.use_imu_yaw_initial_guess = self.use_imu_rotation_initial_guess

        # Scan-to-scan predictor. This never publishes a pose and never
        # inserts keyframes; it only advances the initial guess used by the
        # primary scan-to-map registration.
        self.scan_to_scan_prediction_enabled = bool(
            self.get_parameter("scan_to_scan_prediction_enabled").value
        )
        self.prediction_max_correspondence_distance = float(
            self.get_parameter(
                "prediction_max_correspondence_distance"
            ).value
        )
        self.prediction_max_iterations = int(
            self.get_parameter("prediction_max_iterations").value
        )
        self.prediction_min_fitness = float(
            self.get_parameter("prediction_min_fitness").value
        )
        self.prediction_max_rmse = float(
            self.get_parameter("prediction_max_rmse").value
        )

        # Relocalisation / recovery. The normal rolling local map remains
        # small and fast. A second, coarser historical map is built lazily
        # from validated keyframes and is used ONLY after consecutive
        # scan-to-map failures. This prevents rejected scans from corrupting
        # the map while still allowing the front-end to recover after a turn.
        self.recovery_enabled = bool(
            self.get_parameter("recovery_enabled").value
        )
        self.recovery_use_optimized_map = bool(
            self.get_parameter("recovery_use_optimized_map").value
        )
        self.recovery_optimized_map_topic = str(
            self.get_parameter("recovery_optimized_map_topic").value
        )
        self.recovery_trigger_rejections = int(
            self.get_parameter("recovery_trigger_rejections").value
        )
        self.recovery_attempt_every_n = max(1, int(
            self.get_parameter("recovery_attempt_every_n").value
        ))
        self.recovery_max_keyframes = max(1, int(
            self.get_parameter("recovery_max_keyframes").value
        ))
        self.recovery_map_voxel_size = float(
            self.get_parameter("recovery_map_voxel_size").value
        )
        self.recovery_source_voxel_size = float(
            self.get_parameter("recovery_source_voxel_size").value
        )
        self.recovery_max_map_points = int(
            self.get_parameter("recovery_max_map_points").value
        )
        self.recovery_max_correspondence_distance = float(
            self.get_parameter("recovery_max_correspondence_distance").value
        )
        self.recovery_max_iterations = int(
            self.get_parameter("recovery_max_iterations").value
        )
        self.recovery_min_fitness = float(
            self.get_parameter("recovery_min_fitness").value
        )
        self.recovery_max_rmse = float(
            self.get_parameter("recovery_max_rmse").value
        )
        self.recovery_max_translation_from_tracking_m = float(
            self.get_parameter(
                "recovery_max_translation_from_tracking_m"
            ).value
        )
        self.recovery_max_rotation_from_tracking_rad = math.radians(
            float(self.get_parameter(
                "recovery_max_rotation_from_tracking_deg"
            ).value)
        )
        self.recovery_yaw_offsets_deg = [
            float(v) for v in self.get_parameter(
                "recovery_yaw_offsets_deg"
            ).value
        ]

        # Full 3-D rotational deskew using per-point Livox timestamps and
        # the three IMU gyroscope components. Linear acceleration is NOT
        # integrated: doing so would require robust gravity/bias estimation
        # and is intentionally left out. Translation, including Z, is
        # estimated geometrically by ICP.
        self.deskew_enabled = bool(
            self.get_parameter("deskew_enabled").value
        )
        self.deskew_use_imu_rotation = bool(
            self.get_parameter("deskew_use_imu_rotation").value
        )
        # Alias retained for old status/config compatibility.
        self.deskew_use_imu_yaw = self.deskew_use_imu_rotation
        self.deskew_max_scan_duration_sec = float(
            self.get_parameter("deskew_max_scan_duration_sec").value
        )
        self.deskew_imu_buffer_sec = float(
            self.get_parameter("deskew_imu_buffer_sec").value
        )
        self.deskew_max_imu_extrapolation_sec = float(
            self.get_parameter("deskew_max_imu_extrapolation_sec").value
        )
        self.deskew_max_imu_gap_sec = float(
            self.get_parameter("deskew_max_imu_gap_sec").value
        )

        # Plausibility gate
        self.plausibility_enabled = bool(
            self.get_parameter("plausibility_enabled").value
        )
        self.max_icp_rmse = float(
            self.get_parameter("plausibility_max_rmse").value
        )
        self.max_translation_step_m = float(
            self.get_parameter("max_translation_step_m").value
        )
        self.max_linear_speed_mps = float(
            self.get_parameter("max_linear_speed_mps").value
        )
        self.translation_margin_m = float(
            self.get_parameter("translation_margin_m").value
        )
        self.max_yaw_step_rad = math.radians(
            float(self.get_parameter("max_yaw_step_deg").value)
        )
        self.max_yaw_rate_rad_s = math.radians(
            float(self.get_parameter("max_yaw_rate_deg_s").value)
        )
        self.yaw_margin_rad = math.radians(
            float(self.get_parameter("yaw_margin_deg").value)
        )
        self.use_imu_yaw_gate = bool(
            self.get_parameter("use_imu_yaw_gate").value
        )
        self.max_imu_yaw_difference_rad = math.radians(
            float(self.get_parameter("max_imu_yaw_difference_deg").value)
        )

        # Motion / mapping
        self.planar_mode = bool(self.get_parameter("planar_mode").value)
        self.enable_aligned_cloud = bool(
            self.get_parameter("enable_aligned_cloud").value
        )
        # Dense aligned clouds and the diagnostic point clouds are useful for
        # debugging, but serialising/transferring tens of thousands of points
        # several times per scan is expensive in Python. Optimised operation
        # therefore publishes a voxelised aligned cloud and disables debug
        # clouds unless explicitly requested in the JSON.
        self.aligned_cloud_dense = bool(
            self.get_parameter("aligned_cloud_dense").value
        )
        self.publish_debug_clouds = bool(
            self.get_parameter("publish_debug_clouds").value
        )
        self.enable_local_map = bool(
            self.get_parameter("enable_local_map").value
        )
        self.use_local_map_for_icp = bool(
            self.get_parameter("use_local_map_for_icp").value
        )
        self.min_local_map_frames = int(
            self.get_parameter("min_local_map_frames").value
        )
        self.local_map_voxel_size = float(
            self.get_parameter("local_map_voxel_size").value
        )
        self.max_local_map_points = int(
            self.get_parameter("max_local_map_points").value
        )
        self.max_local_map_keyframes = int(
            self.get_parameter("max_local_map_keyframes").value
        )
        self.keyframe_min_translation = float(
            self.get_parameter("keyframe_min_translation").value
        )
        self.keyframe_min_yaw = math.radians(
            float(self.get_parameter("keyframe_min_yaw_deg").value)
        )

        # Point filtering
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.max_points_before_downsample = int(
            self.get_parameter("max_points_before_downsample").value
        )
        self.min_points = int(self.get_parameter("min_points").value)

        # Debug / outputs
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.publish_status = bool(self.get_parameter("publish_status").value)
        self.max_path_length = int(self.get_parameter("max_path_length").value)

        # Performance / real-time behaviour
        self.expected_scan_period_sec = float(
            self.get_parameter("expected_scan_period_sec").value
        )
        self.drop_detection_factor = float(
            self.get_parameter("drop_detection_factor").value
        )
        self.max_pending_lidar_scans = max(
            1, int(self.get_parameter("max_pending_lidar_scans").value)
        )

        # ============================================================
        # LIVOX MOUNT CORRECTION
        # ============================================================

        self.apply_livox_mount_correction = bool(
            self.cfg("sensors.livox.apply_mount_correction", False)
        )
        self.livox_mount_translation = np.array(
            [
                self.cfg("sensors.livox.translation_m.x", 0.0),
                self.cfg("sensors.livox.translation_m.y", 0.0),
                self.cfg("sensors.livox.translation_m.z", 0.0),
            ],
            dtype=np.float64,
        )
        self.livox_mount_rotation_deg = np.array(
            [
                self.cfg("sensors.livox.rotation_deg.roll", 0.0),
                self.cfg("sensors.livox.rotation_deg.pitch", 0.0),
                self.cfg("sensors.livox.rotation_deg.yaw", 0.0),
            ],
            dtype=np.float64,
        )
        self.livox_mount_transform = self.build_transform_from_xyz_rpy(
            self.livox_mount_translation,
            np.deg2rad(self.livox_mount_rotation_deg),
        )

        self.output_frame = (
            self.corrected_lidar_frame
            if self.apply_livox_mount_correction
            else self.robot_frame
        )

        # ============================================================
        # STATE
        # ============================================================

        # Validated/public pose. Only a pose that passes the primary ICP
        # quality checks and plausibility gate is allowed to modify this.
        self.pose_matrix = np.eye(4, dtype=np.float64)

        # Auxiliary tracking pose. It is advanced on every valid LiDAR scan
        # using scan-to-scan translation + IMU yaw, even if the primary
        # scan-to-map pose is rejected. It is never published directly.
        self.tracking_guess_pose = np.eye(4, dtype=np.float64)

        # Previous processed scan, regardless of whether the primary pose was
        # accepted. This keeps scan-to-scan prediction truly consecutive.
        self.prev_scan_cloud = None
        self.prev_scan_stamp = None
        self.prev_scan_imu_rotation = None

        # Last VALIDATED pose timestamp/IMU orientation. These are kept
        # separate from the previous processed scan because the plausibility
        # gate must compare against the last pose actually published.
        self.last_accepted_stamp = None
        self.last_accepted_imu_rotation = None

        self.cloud_counter = 0
        self.last_lidar_stamp_sec = None
        self.last_processed_lidar_stamp_sec = None
        self.reset_counter = 0
        self.consecutive_rejections = 0

        # Estimated input loss / overload diagnostics.
        self.estimated_dropped_scans = 0
        self.pending_lidar_queue_drops = 0

        # Fine-grained profiling for each processed LiDAR scan.
        self.current_profile_preprocess_ms = 0.0
        self.current_profile_deskew_ms = 0.0
        self.current_profile_icp_ms = 0.0
        self.current_profile_icp_calls = 0

        # Shared state is accessed from separate ROS callback groups.
        self.imu_state_lock = threading.RLock()
        self.pending_lidar_lock = threading.Lock()
        self.processing_lidar_lock = threading.Lock()
        self.recovery_map_lock = threading.RLock()

        # Integrated orientation from the gyroscope. This is not published
        # as odometry; it is only used for ICP initial guesses and deskew.
        self.imu_rotation_matrix = np.eye(3, dtype=np.float64)
        self.imu_yaw = 0.0  # diagnostic/backward-compatible view only
        self.last_imu_time = None

        # Corrected 3-D gyro history used for per-point rotational deskew.
        # Entries are (timestamp_sec, wx, wy, wz), all in the corrected
        # LiDAR/body frame and in rad/s.
        self.imu_gyro_buffer = deque()
        # Absolute integrated IMU orientation snapshots, used to associate a
        # delayed LiDAR scan with the IMU orientation at the scan timestamp
        # rather than with whatever IMU sample happened to be newest when the
        # queued scan is finally processed.
        self.imu_orientation_buffer = deque()
        self.last_deskew_status = self.empty_deskew_status()

        # LiDAR scans are delayed until the IMU buffer covers the complete
        # acquisition interval. This avoids alternating between deskewed and
        # non-deskewed scans simply because the last IMU samples have not
        # arrived yet when the LiDAR callback fires.
        self.pending_lidar_scans = deque()
        self.processing_pending_lidar = False

        # Rotate one recovery yaw seed per attempt instead of evaluating all
        # hypotheses in one callback. This bounds worst-case recovery latency.
        self.recovery_yaw_seed_index = 0

        # Recovery fallback: several consecutive valid scan-to-scan
        # predictions are required before the auxiliary tracking pose may be
        # promoted back to the validated/public pose.
        self.recovery_tracking_valid_streak = 0
        self.recovery_tracking_confirmations_required = 3

        # Rolling local map: the deque is the ONLY source of local-map data.
        rolling_window = max(1, self.max_local_map_keyframes)
        self.local_map_keyframes = deque(maxlen=rolling_window)
        self.local_map_points = None
        self.local_map_cloud = None
        self.local_map_frame_count = 0
        self.keyframe_count = 0
        self.last_keyframe_pose = None

        # Historical validated keyframes for relocalisation. This map is
        # intentionally separate from the rolling local map. It is built
        # lazily only when recovery is needed, so healthy tracking pays
        # almost no extra CPU cost.
        self.recovery_map_keyframes = deque(
            maxlen=self.recovery_max_keyframes
        )
        self.recovery_map_points = None
        self.recovery_map_cloud = None
        self.recovery_map_dirty = True
        self.latest_optimized_map_msg = None
        self.optimized_recovery_map_points = None
        self.optimized_recovery_map_cloud = None
        self.current_recovery_map_source = "historical_frontend"
        self.last_recovery_status = self.empty_recovery_status()

        self.path_msg = PathMsg()
        self.path_msg.header.frame_id = self.fixed_frame

        # ============================================================
        # ROS INTERFACE
        # ============================================================

        # IMU reception must remain responsive while Open3D ICP is running.
        # LiDAR reception is also kept lightweight; heavy scan processing is
        # performed by a dedicated callback group.
        self.imu_callback_group = MutuallyExclusiveCallbackGroup()
        self.lidar_callback_group = MutuallyExclusiveCallbackGroup()
        self.processing_callback_group = MutuallyExclusiveCallbackGroup()
        self.recovery_map_callback_group = MutuallyExclusiveCallbackGroup()

        self.pose_pub = self.create_publisher(PoseStamped, "/g1/slam/pose", 10)
        self.odom_pub = self.create_publisher(Odometry, "/g1/slam/odom", 10)
        self.path_pub = self.create_publisher(PathMsg, "/g1/slam/path", 10)
        self.status_pub = self.create_publisher(String, "/g1/slam/status", 10)
        self.marker_pub = self.create_publisher(
            Marker, "/g1/slam/robot_marker", 10
        )
        self.aligned_cloud_pub = self.create_publisher(
            PointCloud2, "/g1/slam/aligned_cloud", 10
        )
        # Diagnostic views of the current LiDAR scan. These are never used
        # by the pose-graph backend or inserted into the local map.
        #
        # raw_aligned_cloud: current filtered raw scan placed with the last
        # validated/public pose. It is published even when ICP rejects the
        # current scan, so loss of tracking is visible instead of freezing.
        #
        # predicted_cloud: the same current scan placed with the auxiliary
        # tracking predictor. Comparing both clouds isolates whether the
        # predictor or the primary scan-to-map registration is failing.
        self.raw_aligned_cloud_pub = self.create_publisher(
            PointCloud2, "/g1/slam/debug/raw_aligned_cloud", 10
        )
        self.predicted_cloud_pub = self.create_publisher(
            PointCloud2, "/g1/slam/debug/predicted_cloud", 10
        )

        # Deskew diagnostics. Both clouds are placed with EXACTLY the same
        # validated pose. The only difference is whether per-point IMU deskew
        # has been applied. This isolates deskew errors from pose-estimation
        # errors.
        self.raw_no_deskew_cloud_pub = self.create_publisher(
            PointCloud2, "/g1/slam/debug/raw_no_deskew_cloud", 10
        )
        self.deskewed_cloud_pub = self.create_publisher(
            PointCloud2, "/g1/slam/debug/deskewed_cloud", 10
        )

        # Current primary ICP hypothesis, published even when the plausibility
        # gate later rejects it. It is diagnostic only and is never inserted
        # into the local map or pose graph.
        self.icp_candidate_cloud_pub = self.create_publisher(
            PointCloud2, "/g1/slam/debug/icp_candidate_cloud", 10
        )

        self.local_map_pub = self.create_publisher(
            PointCloud2, "/g1/slam/local_map", 10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
            callback_group=self.imu_callback_group,
        )

        # Explicit latest-only sensor QoS: if the front-end cannot keep up,
        # stale LiDAR samples must not accumulate in the DDS subscription.
        lidar_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            lidar_qos,
            callback_group=self.lidar_callback_group,
        )

        # Pending deskew scans are released outside the IMU callback. This is
        # important with a multithreaded executor: the IMU callback remains
        # cheap and continues filling the temporal buffer while ICP runs.
        self.pending_lidar_timer = self.create_timer(
            0.005,
            self.process_ready_pending_lidar,
            callback_group=self.processing_callback_group,
        )

        self.optimized_map_sub = None
        if self.recovery_enabled and self.recovery_use_optimized_map:
            optimized_map_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.optimized_map_sub = self.create_subscription(
                PointCloud2,
                self.recovery_optimized_map_topic,
                self.optimized_map_callback,
                optimized_map_qos,
                callback_group=self.recovery_map_callback_group,
            )

        self.get_logger().info("Livox custom ICP node started")
        self.get_logger().info(
            "ICP target policy: previous scan during bootstrap, then rolling local map"
        )
        self.get_logger().info(
            "Rolling local map: "
            f"last {self.max_local_map_keyframes} keyframes, "
            f"max {self.max_local_map_points} points"
        )
        self.get_logger().info(
            f"IMU 3-D rotation initial guess: {self.use_imu_rotation_initial_guess}; "
            f"plausibility gate: {self.plausibility_enabled}"
        )
        self.get_logger().info(
            "Scan-to-scan tracking predictor: "
            f"{self.scan_to_scan_prediction_enabled}"
        )
        self.get_logger().info(
            "LiDAR 3-D rotational deskew: "
            f"{self.deskew_enabled and self.deskew_use_imu_rotation}"
        )
        self.get_logger().info(
            "Optimised output mode: "
            f"aligned_cloud_dense={self.aligned_cloud_dense}, "
            f"debug_clouds={self.publish_debug_clouds}"
        )
        self.get_logger().info(
            "Relocalisation recovery: "
            f"enabled={self.recovery_enabled}, "
            f"trigger={self.recovery_trigger_rejections}, "
            f"historical_keyframes={self.recovery_max_keyframes}, "
            f"prefer_optimized_map={self.recovery_use_optimized_map}"
        )
        self.get_logger().info(
            "Realtime execution: MultiThreadedExecutor(2), "
            "LiDAR QoS KEEP_LAST(1), separate IMU/LiDAR/processing callbacks"
        )

    # ============================================================
    # CONFIG HELPERS
    # ============================================================

    def declare_runtime_parameters(self):
        self.declare_parameter(
            "lidar_topic", self.cfg("topics.lidar", "/livox/lidar")
        )
        self.declare_parameter(
            "imu_topic", self.cfg("topics.imu", "/livox/imu")
        )
        self.declare_parameter(
            "fixed_frame", self.cfg("frames.fixed_frame", "map")
        )
        self.declare_parameter(
            "robot_frame", self.cfg("frames.robot_frame", "livox_frame")
        )
        self.declare_parameter(
            "corrected_lidar_frame",
            self.cfg("frames.corrected_lidar_frame", "livox_corrected_frame"),
        )

        self.declare_parameter("voxel_size", self.icp_cfg("voxel_size", 0.15))
        self.declare_parameter(
            "icp_max_correspondence_distance",
            self.icp_cfg("max_correspondence_distance", 1.0),
        )
        self.declare_parameter(
            "icp_max_iterations", self.icp_cfg("max_iterations", 30)
        )
        self.declare_parameter(
            "min_fitness", self.icp_cfg("min_fitness", 0.15)
        )
        self.declare_parameter("cloud_skip", self.icp_cfg("cloud_skip", 1))
        self.declare_parameter(
            "use_imu_rotation_initial_guess",
            self.icp_cfg(
                "use_imu_rotation_initial_guess",
                self.icp_cfg("use_imu_yaw_initial_guess", True),
            ),
        )

        self.declare_parameter(
            "scan_to_scan_prediction_enabled",
            self.icp_cfg("scan_to_scan_prediction.enabled", True),
        )
        self.declare_parameter(
            "prediction_max_correspondence_distance",
            self.icp_cfg(
                "scan_to_scan_prediction.max_correspondence_distance",
                0.7,
            ),
        )
        self.declare_parameter(
            "prediction_max_iterations",
            self.icp_cfg("scan_to_scan_prediction.max_iterations", 15),
        )
        self.declare_parameter(
            "prediction_min_fitness",
            self.icp_cfg("scan_to_scan_prediction.min_fitness", 0.15),
        )
        self.declare_parameter(
            "prediction_max_rmse",
            self.icp_cfg("scan_to_scan_prediction.max_rmse", 0.35),
        )

        self.declare_parameter(
            "recovery_enabled",
            self.icp_cfg("recovery.enabled", True),
        )
        self.declare_parameter(
            "recovery_use_optimized_map",
            self.icp_cfg("recovery.use_optimized_map", True),
        )
        self.declare_parameter(
            "recovery_optimized_map_topic",
            self.icp_cfg(
                "recovery.optimized_map_topic",
                "/g1/slam/optimized/global_map",
            ),
        )
        self.declare_parameter(
            "recovery_trigger_rejections",
            self.icp_cfg("recovery.trigger_after_rejections", 3),
        )
        self.declare_parameter(
            "recovery_attempt_every_n",
            self.icp_cfg("recovery.attempt_every_n_rejections", 2),
        )
        self.declare_parameter(
            "recovery_max_keyframes",
            self.icp_cfg("recovery.max_keyframes", 250),
        )
        self.declare_parameter(
            "recovery_map_voxel_size",
            self.icp_cfg("recovery.map_voxel_size", 0.25),
        )
        self.declare_parameter(
            "recovery_source_voxel_size",
            self.icp_cfg("recovery.source_voxel_size", 0.20),
        )
        self.declare_parameter(
            "recovery_max_map_points",
            self.icp_cfg("recovery.max_map_points", 60000),
        )
        self.declare_parameter(
            "recovery_max_correspondence_distance",
            self.icp_cfg("recovery.max_correspondence_distance", 1.5),
        )
        self.declare_parameter(
            "recovery_max_iterations",
            self.icp_cfg("recovery.max_iterations", 15),
        )
        self.declare_parameter(
            "recovery_min_fitness",
            self.icp_cfg("recovery.min_fitness", 0.30),
        )
        self.declare_parameter(
            "recovery_max_rmse",
            self.icp_cfg("recovery.max_rmse", 0.35),
        )
        self.declare_parameter(
            "recovery_max_translation_from_tracking_m",
            self.icp_cfg(
                "recovery.max_translation_from_tracking_m", 2.0
            ),
        )
        self.declare_parameter(
            "recovery_max_rotation_from_tracking_deg",
            self.icp_cfg(
                "recovery.max_rotation_from_tracking_deg", 60.0
            ),
        )
        self.declare_parameter(
            "recovery_yaw_offsets_deg",
            self.icp_cfg("recovery.yaw_offsets_deg", [0.0, -20.0, 20.0]),
        )

        self.declare_parameter(
            "deskew_enabled",
            self.icp_cfg("deskew.enabled", True),
        )
        self.declare_parameter(
            "deskew_use_imu_rotation",
            self.icp_cfg(
                "deskew.use_imu_rotation",
                self.icp_cfg("deskew.use_imu_yaw", True),
            ),
        )
        self.declare_parameter(
            "deskew_max_scan_duration_sec",
            self.icp_cfg("deskew.max_scan_duration_sec", 0.15),
        )
        self.declare_parameter(
            "deskew_imu_buffer_sec",
            self.icp_cfg("deskew.imu_buffer_sec", 2.0),
        )
        self.declare_parameter(
            "deskew_max_imu_extrapolation_sec",
            self.icp_cfg("deskew.max_imu_extrapolation_sec", 0.02),
        )
        self.declare_parameter(
            "deskew_max_imu_gap_sec",
            self.icp_cfg("deskew.max_imu_gap_sec", 0.05),
        )

        self.declare_parameter(
            "plausibility_enabled",
            self.icp_cfg("plausibility.enabled", True),
        )
        self.declare_parameter(
            "plausibility_max_rmse",
            self.icp_cfg("plausibility.max_rmse", 0.30),
        )
        self.declare_parameter(
            "max_translation_step_m",
            self.icp_cfg("plausibility.max_translation_step_m", 1.5),
        )
        self.declare_parameter(
            "max_linear_speed_mps",
            self.icp_cfg("plausibility.max_linear_speed_mps", 3.0),
        )
        self.declare_parameter(
            "translation_margin_m",
            self.icp_cfg("plausibility.translation_margin_m", 0.15),
        )
        self.declare_parameter(
            "max_yaw_step_deg",
            self.icp_cfg("plausibility.max_yaw_step_deg", 120.0),
        )
        self.declare_parameter(
            "max_yaw_rate_deg_s",
            self.icp_cfg("plausibility.max_yaw_rate_deg_s", 360.0),
        )
        self.declare_parameter(
            "yaw_margin_deg",
            self.icp_cfg("plausibility.yaw_margin_deg", 5.0),
        )
        self.declare_parameter(
            "use_imu_yaw_gate",
            self.icp_cfg("plausibility.use_imu_yaw_gate", True),
        )
        self.declare_parameter(
            "max_imu_yaw_difference_deg",
            self.icp_cfg("plausibility.max_imu_yaw_difference_deg", 12.0),
        )

        self.declare_parameter(
            "enable_aligned_cloud",
            self.cfg("mapping.enable_aligned_cloud", True),
        )
        self.declare_parameter(
            "aligned_cloud_dense",
            self.cfg("mapping.aligned_cloud_dense", False),
        )
        self.declare_parameter(
            "publish_debug_clouds",
            self.cfg("debug.publish_clouds", False),
        )
        self.declare_parameter(
            "enable_local_map",
            self.cfg("mapping.enable_local_map", True),
        )
        self.declare_parameter(
            "use_local_map_for_icp",
            self.cfg("mapping.use_local_map_for_icp", True),
        )
        self.declare_parameter(
            "min_local_map_frames",
            self.cfg("mapping.min_local_map_frames", 3),
        )
        self.declare_parameter(
            "local_map_voxel_size",
            self.cfg("mapping.local_map_voxel_size", 0.15),
        )
        self.declare_parameter(
            "max_local_map_points",
            self.cfg("mapping.max_local_map_points", 30000),
        )
        self.declare_parameter(
            "max_local_map_keyframes",
            self.cfg("mapping.max_local_map_keyframes", 12),
        )
        self.declare_parameter(
            "keyframe_min_translation",
            self.cfg("mapping.keyframe_min_translation", 0.25),
        )
        self.declare_parameter(
            "keyframe_min_yaw_deg",
            self.cfg("mapping.keyframe_min_yaw_deg", 7.0),
        )

        self.declare_parameter(
            "planar_mode", self.cfg("motion_model.planar_mode", True)
        )

        self.declare_parameter("min_range", self.cfg("filters.min_range", 0.4))
        self.declare_parameter("max_range", self.cfg("filters.max_range", 35.0))
        self.declare_parameter(
            "max_points_before_downsample",
            self.cfg("filters.max_points_before_downsample", 120000),
        )
        self.declare_parameter("min_points", self.cfg("filters.min_points", 800))

        self.declare_parameter(
            "publish_tf", self.cfg("debug.publish_tf", True)
        )
        self.declare_parameter(
            "publish_status", self.cfg("debug.publish_status", True)
        )
        self.declare_parameter(
            "max_path_length", self.cfg("debug.max_path_length", 5000)
        )

        self.declare_parameter(
            "expected_scan_period_sec",
            self.cfg("performance.expected_scan_period_sec", 0.1),
        )
        self.declare_parameter(
            "drop_detection_factor",
            self.cfg("performance.drop_detection_factor", 1.5),
        )
        self.declare_parameter(
            "max_pending_lidar_scans",
            self.cfg("performance.max_pending_lidar_scans", 2),
        )

    def load_json_config(self, config_path: str) -> dict:
        if not config_path:
            return {}

        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            self.get_logger().warn(f"Config file not found: {path}")
            return {}

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def cfg(self, key: str, default: Any) -> Any:
        value = self.config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def icp_cfg(self, key: str, default: Any) -> Any:
        legacy = self.cfg(f"icp.{key}", default)
        return self.cfg(f"slam.algorithm_options.icp.{key}", legacy)

    # ============================================================
    # IMU
    # ============================================================

    def imu_callback(self, msg: Imu):
        """Integrate the gyroscope while keeping the callback lightweight.

        Heavy LiDAR/ICP work is deliberately not executed here. The dedicated
        processing timer consumes scans once this callback has supplied enough
        IMU coverage for deskew.
        """
        current_time = self.stamp_to_seconds(msg.header.stamp)

        angular_velocity = np.array(
            [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ],
            dtype=np.float64,
        )

        if self.apply_livox_mount_correction:
            angular_velocity = (
                self.livox_mount_transform[:3, :3] @ angular_velocity
            )

        if not np.all(np.isfinite(angular_velocity)):
            return

        with self.imu_state_lock:
            self.imu_gyro_buffer.append(
                (
                    current_time,
                    float(angular_velocity[0]),
                    float(angular_velocity[1]),
                    float(angular_velocity[2]),
                )
            )

            cutoff = current_time - max(0.5, self.deskew_imu_buffer_sec)
            while (
                len(self.imu_gyro_buffer) > 2
                and self.imu_gyro_buffer[0][0] < cutoff
            ):
                self.imu_gyro_buffer.popleft()

            if self.last_imu_time is None:
                self.last_imu_time = current_time
                self.imu_orientation_buffer.append(
                    (current_time, self.imu_rotation_matrix.copy())
                )
                return

            dt = current_time - self.last_imu_time
            self.last_imu_time = current_time

            if dt <= 0.0 or dt > 1.0:
                return

            delta_rotation = self.rotation_vector_to_matrix(
                angular_velocity * dt
            )
            self.imu_rotation_matrix = (
                self.imu_rotation_matrix @ delta_rotation
            )

            u, _, vt = np.linalg.svd(self.imu_rotation_matrix)
            self.imu_rotation_matrix = u @ vt
            if np.linalg.det(self.imu_rotation_matrix) < 0.0:
                u[:, -1] *= -1.0
                self.imu_rotation_matrix = u @ vt

            self.imu_yaw = self.get_yaw_from_rotation(
                self.imu_rotation_matrix
            )

            self.imu_orientation_buffer.append(
                (current_time, self.imu_rotation_matrix.copy())
            )
            orientation_cutoff = (
                current_time - max(0.5, self.deskew_imu_buffer_sec)
            )
            while (
                len(self.imu_orientation_buffer) > 2
                and self.imu_orientation_buffer[0][0] < orientation_cutoff
            ):
                self.imu_orientation_buffer.popleft()

    def get_imu_rotation_at_time(
        self, timestamp_sec: float
    ) -> Optional[np.ndarray]:
        """Return the nearest integrated IMU orientation to a scan timestamp."""
        with self.imu_state_lock:
            if not self.imu_orientation_buffer:
                return None

            entries = list(self.imu_orientation_buffer)

        times = np.fromiter(
            (entry[0] for entry in entries),
            dtype=np.float64,
        )
        if times.size == 0:
            return None

        idx = int(np.searchsorted(times, timestamp_sec))
        if idx <= 0:
            chosen = 0
        elif idx >= len(times):
            chosen = len(times) - 1
        else:
            before = idx - 1
            after = idx
            chosen = (
                before
                if abs(timestamp_sec - times[before])
                <= abs(times[after] - timestamp_sec)
                else after
            )

        if (
            self.deskew_max_imu_gap_sec > 0.0
            and abs(float(times[chosen]) - float(timestamp_sec))
            > self.deskew_max_imu_gap_sec
        ):
            return None

        return entries[chosen][1].copy()

    def lidar_callback(self, msg: PointCloud2):
        """Keep LiDAR reception cheap and queue only a small latest window."""
        if self.deskew_enabled and self.deskew_use_imu_rotation:
            scan_end_time = self.get_scan_end_time_from_msg(msg)
            if scan_end_time is not None:
                with self.pending_lidar_lock:
                    while (
                        len(self.pending_lidar_scans)
                        >= self.max_pending_lidar_scans
                    ):
                        self.pending_lidar_scans.popleft()
                        self.pending_lidar_queue_drops += 1
                    self.pending_lidar_scans.append((scan_end_time, msg))
                return

        # No deskew/timestamp: still process in the dedicated processing group
        # by using the header time as an already-ready queue item.
        with self.pending_lidar_lock:
            while (
                len(self.pending_lidar_scans)
                >= self.max_pending_lidar_scans
            ):
                self.pending_lidar_scans.popleft()
                self.pending_lidar_queue_drops += 1
            self.pending_lidar_scans.append(
                (self.stamp_to_seconds(msg.header.stamp), msg)
            )

    def process_ready_pending_lidar(self):
        """Process at most one ready scan per timer invocation."""
        if not self.processing_lidar_lock.acquire(blocking=False):
            return

        try:
            with self.imu_state_lock:
                latest_imu_time = self.last_imu_time

            with self.pending_lidar_lock:
                if not self.pending_lidar_scans:
                    return

                scan_end_time, msg = self.pending_lidar_scans[0]

                if (
                    self.deskew_enabled
                    and self.deskew_use_imu_rotation
                    and (
                        latest_imu_time is None
                        or latest_imu_time + 1e-9 < scan_end_time
                    )
                ):
                    return

                self.pending_lidar_scans.popleft()

            self.process_lidar_message(msg)
        finally:
            self.processing_lidar_lock.release()

    def get_scan_end_time_from_msg(self, msg: PointCloud2):
        """Read the last Livox point timestamp without deserialising XYZ.

        The TFM bags store ``timestamp`` as FLOAT64 absolute nanoseconds.
        Returning ``None`` keeps the code safe for heterogeneous PointCloud2
        messages and lets the caller use the legacy immediate path.
        """
        import struct

        timestamp_field = next(
            (field for field in msg.fields if field.name == "timestamp"),
            None,
        )
        if timestamp_field is None:
            return None

        # sensor_msgs/PointField.FLOAT64 == 8
        if int(timestamp_field.datatype) != 8:
            return None
        if msg.width <= 0 or msg.height <= 0 or msg.point_step <= 0:
            return None

        try:
            row = int(msg.height) - 1
            col = int(msg.width) - 1
            byte_offset = (
                row * int(msg.row_step)
                + col * int(msg.point_step)
                + int(timestamp_field.offset)
            )
            fmt = ">d" if msg.is_bigendian else "<d"
            timestamp_ns = struct.unpack_from(
                fmt, msg.data, byte_offset
            )[0]
        except (IndexError, TypeError, ValueError, struct.error):
            return None

        if not math.isfinite(timestamp_ns):
            return None

        header_time = self.stamp_to_seconds(msg.header.stamp)
        point_time = float(timestamp_ns) * 1e-9

        # Reject clearly incompatible timestamp conventions.
        if point_time < header_time - 0.01:
            return None
        if (
            self.deskew_max_scan_duration_sec > 0.0
            and point_time - header_time
            > self.deskew_max_scan_duration_sec + 0.01
        ):
            return None

        return point_time

    def process_lidar_message(self, msg: PointCloud2):
        """Process one Livox scan with a single-ICP healthy fast path.

        Bootstrap still uses scan-to-scan because no useful local map exists.
        After the rolling map is ready, the normal path is:

            previous validated/tracking pose + IMU rotation increment
                -> one scan-to-map ICP
                -> accept and re-anchor

        Only when that primary registration fails do we run the more expensive
        scan-to-scan predictor. The predictor advances ``tracking_guess_pose``
        for the next frame but never directly contaminates the map.

        This is intentionally structured to keep the healthy case close to one
        Open3D ICP call per LiDAR scan instead of two.
        """
        start_time = time.perf_counter()
        msg_time = self.stamp_to_seconds(msg.header.stamp)
        self.last_recovery_status = self.empty_recovery_status()

        self.current_profile_preprocess_ms = 0.0
        self.current_profile_deskew_ms = 0.0
        self.current_profile_icp_ms = 0.0
        self.current_profile_icp_calls = 0

        if (
            self.last_processed_lidar_stamp_sec is not None
            and self.expected_scan_period_sec > 0.0
        ):
            scan_gap = msg_time - self.last_processed_lidar_stamp_sec
            if (
                scan_gap
                > self.expected_scan_period_sec
                * self.drop_detection_factor
            ):
                estimated = max(
                    0,
                    int(round(scan_gap / self.expected_scan_period_sec)) - 1,
                )
                self.estimated_dropped_scans += estimated
        self.last_processed_lidar_stamp_sec = msg_time

        # Because deskewed scans may be processed ~100 ms after their header
        # timestamp, never use the newest global IMU orientation as the scan
        # orientation. Associate this scan with its own timestamp instead.
        scan_imu_rotation = self.get_imu_rotation_at_time(msg_time)
        if scan_imu_rotation is None:
            with self.imu_state_lock:
                scan_imu_rotation = self.imu_rotation_matrix.copy()

        if (
            self.enable_auto_reset_on_clock_jump
            and self.last_lidar_stamp_sec is not None
            and msg_time
            < self.last_lidar_stamp_sec - self.clock_jump_threshold_sec
        ):
            self.get_logger().warn(
                "Detected bag loop/reset. Resetting SLAM state."
            )
            self.reset_slam_state(msg.header.stamp)

        self.last_lidar_stamp_sec = msg_time
        self.cloud_counter += 1

        if self.cloud_skip > 1 and self.cloud_counter % self.cloud_skip != 0:
            return

        if self.deskew_enabled and self.deskew_use_imu_rotation:
            points, point_timestamps_ns = (
                self.pointcloud2_to_xyz_timestamp_array(msg)
            )
        else:
            # Fast path when deskew is disabled: do not deserialize the
            # per-point FLOAT64 timestamp field unnecessarily.
            points = self.pointcloud2_to_xyz_array(msg)
            point_timestamps_ns = None

        self.last_deskew_status = self.empty_deskew_status()
        self.last_deskew_status["timestamp_available"] = (
            point_timestamps_ns is not None
        )

        if points.shape[0] < self.min_points:
            self.reject_without_candidate(
                msg.header.stamp,
                start_time,
                "not_enough_raw_points",
                int(points.shape[0]),
            )
            return

        # Mount rotation first so LiDAR and gyro use the same corrected basis.
        if self.apply_livox_mount_correction:
            points = points @ self.livox_mount_transform[:3, :3].T

        # Only keep the no-deskew duplicate when the user explicitly enables
        # debug clouds. In normal operation this avoids a full dense copy.
        raw_no_deskew_points = (
            points.copy() if self.publish_debug_clouds else None
        )

        deskew_start = time.perf_counter()
        points, deskew_status = self.deskew_points_3d(
            points=points,
            point_timestamps_ns=point_timestamps_ns,
            scan_stamp=msg.header.stamp,
        )
        self.current_profile_deskew_ms = (
            time.perf_counter() - deskew_start
        ) * 1000.0
        self.last_deskew_status = deskew_status

        if self.apply_livox_mount_correction:
            mount_translation = self.livox_mount_transform[:3, 3]
            points = points + mount_translation
            if raw_no_deskew_points is not None:
                raw_no_deskew_points = (
                    raw_no_deskew_points + mount_translation
                )

        # Debug comparison needs identical samples in both clouds. When debug
        # is disabled, filter only the actual ICP cloud to avoid extra work.
        if raw_no_deskew_points is not None:
            common_range_mask = self.range_filter_mask(
                raw_no_deskew_points
            )
            raw_no_deskew_points = raw_no_deskew_points[
                common_range_mask
            ]
            points = points[common_range_mask]
        else:
            points = self.filter_points_by_range(points)

        if (
            self.max_points_before_downsample > 0
            and points.shape[0] > self.max_points_before_downsample
        ):
            limit_indices = np.linspace(
                0,
                points.shape[0] - 1,
                num=self.max_points_before_downsample,
                dtype=np.int64,
            )
            points = points[limit_indices]
            if raw_no_deskew_points is not None:
                raw_no_deskew_points = raw_no_deskew_points[
                    limit_indices
                ]

        # Dense pre-voxel points are only retained if requested for visual
        # output. The normal aligned cloud can use the same voxelised scan that
        # ICP already created, cutting PointCloud2 serialisation substantially.
        need_dense_display = (
            self.aligned_cloud_dense or self.publish_debug_clouds
        )
        display_points = points.copy() if need_dense_display else None
        raw_no_deskew_display_points = (
            raw_no_deskew_points.copy()
            if raw_no_deskew_points is not None
            else None
        )

        cloud = self.create_open3d_cloud(points)
        if cloud is None:
            self.reject_without_candidate(
                msg.header.stamp,
                start_time,
                "invalid_cloud_after_filtering",
                int(points.shape[0]),
            )
            return

        current_points = np.asarray(cloud.points)
        aligned_source_points = (
            display_points
            if self.aligned_cloud_dense and display_points is not None
            else current_points
        )
        debug_source_points = (
            display_points
            if display_points is not None
            else current_points
        )

        self.current_profile_preprocess_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # --------------------------------------------------------
        # First accepted frame
        # --------------------------------------------------------

        if self.prev_scan_cloud is None:
            self.pose_matrix = np.eye(4, dtype=np.float64)
            self.tracking_guess_pose = self.pose_matrix.copy()

            self.prev_scan_cloud = cloud
            self.prev_scan_stamp = msg.header.stamp
            self.prev_scan_imu_rotation = scan_imu_rotation.copy()

            self.last_accepted_stamp = msg.header.stamp
            self.last_accepted_imu_rotation = scan_imu_rotation.copy()
            self.consecutive_rejections = 0

            aligned_points = self.transform_points_to_map(current_points)
            keyframe_added = self.try_add_keyframe(aligned_points)

            aligned_output = self.apply_transform_to_points(
                aligned_source_points, self.pose_matrix
            )
            self.publish_aligned_cloud(
                aligned_output, msg.header.stamp
            )

            if self.publish_debug_clouds:
                self.publish_diagnostic_clouds(
                    debug_source_points, msg.header.stamp
                )
                self.publish_deskew_comparison_clouds(
                    raw_no_deskew_display_points,
                    debug_source_points,
                    msg.header.stamp,
                )

            if keyframe_added:
                self.publish_local_map(msg.header.stamp)
            self.publish_outputs(msg.header.stamp)

            self.publish_status_msg(
                accepted=True,
                reason="initial_frame",
                stamp=msg.header.stamp,
                points=int(current_points.shape[0]),
                fitness=None,
                rmse=None,
                processing_ms=self.elapsed_ms(start_time),
                icp_target="initial",
                keyframe_added=keyframe_added,
                gate=self.zero_gate(),
                prediction=self.empty_prediction_status(),
            )
            return

        # --------------------------------------------------------
        # IMU increments
        # --------------------------------------------------------

        scan_imu_rotation_delta = None
        if (
            self.use_imu_rotation_initial_guess
            and self.prev_scan_imu_rotation is not None
        ):
            scan_imu_rotation_delta = (
                self.prev_scan_imu_rotation.T
                @ scan_imu_rotation
            )

        accepted_imu_rotation_delta = None
        if (
            self.use_imu_rotation_initial_guess
            and self.last_accepted_imu_rotation is not None
        ):
            accepted_imu_rotation_delta = (
                self.last_accepted_imu_rotation.T
                @ scan_imu_rotation
            )

        use_local_map = (
            self.enable_local_map
            and self.use_local_map_for_icp
            and self.local_map_cloud is not None
            and self.local_map_frame_count >= self.min_local_map_frames
        )

        prediction = self.empty_prediction_status()
        candidate_pose = None
        fitness = None
        rmse = None
        gate = None

        # --------------------------------------------------------
        # RECOVERY MODE
        # --------------------------------------------------------

        recovery_active = (
            self.recovery_enabled
            and use_local_map
            and self.consecutive_rejections
            >= self.recovery_trigger_rejections
            and len(self.recovery_map_keyframes) >= self.min_local_map_frames
        )

        if recovery_active:
            self.last_recovery_status["active"] = True
            recovery_index = (
                self.consecutive_rejections
                - self.recovery_trigger_rejections
            )
            attempt_relocalisation = (
                recovery_index % self.recovery_attempt_every_n == 0
            )

            if attempt_relocalisation:
                recovery = self.try_relocalize(
                    cloud=cloud,
                    tracking_guess=self.tracking_guess_pose,
                )
                self.last_recovery_status = recovery["status"]
                candidate_pose = recovery["pose"]

                # The current scan always becomes the consecutive reference
                # for the next recovery-tracking frame.
                self.prev_scan_cloud = cloud
                self.prev_scan_stamp = msg.header.stamp
                self.prev_scan_imu_rotation = scan_imu_rotation.copy()

                if candidate_pose is not None:
                    fitness = recovery["status"]["fitness"]
                    rmse = recovery["status"]["rmse"]

                    recovery_gate = self.build_recovery_gate_for_status(
                        self.pose_matrix,
                        candidate_pose,
                        msg.header.stamp,
                        accepted_imu_rotation_delta,
                    )

                    self.pose_matrix = candidate_pose
                    self.tracking_guess_pose = candidate_pose.copy()
                    self.last_accepted_stamp = msg.header.stamp
                    self.last_accepted_imu_rotation = (
                        scan_imu_rotation.copy()
                    )
                    self.consecutive_rejections = 0
                    self.recovery_tracking_valid_streak = 0

                    aligned_points = self.transform_points_to_map(
                        current_points
                    )
                    keyframe_added = self.try_add_keyframe(aligned_points)

                    aligned_output = self.apply_transform_to_points(
                        aligned_source_points, self.pose_matrix
                    )
                    self.publish_aligned_cloud(
                        aligned_output, msg.header.stamp
                    )
                    if keyframe_added:
                        self.publish_local_map(msg.header.stamp)
                    self.publish_outputs(msg.header.stamp)

                    self.publish_status_msg(
                        accepted=True,
                        reason="relocalized",
                        stamp=msg.header.stamp,
                        points=int(current_points.shape[0]),
                        fitness=fitness,
                        rmse=rmse,
                        processing_ms=self.elapsed_ms(start_time),
                        icp_target="recovery_map",
                        keyframe_added=keyframe_added,
                        gate=recovery_gate,
                        prediction=prediction,
                    )
                    return

                # A failed relocalisation frame does not run a second ICP.
                # Keep only the accumulated IMU rotation alive and alternate
                # with scan-to-scan tracking on the next recovery frame.
                self.advance_tracking_rotation_only(
                    scan_imu_rotation_delta
                )
                self.consecutive_rejections += 1
                recovery_gate = self.empty_gate(
                    msg.header.stamp, accepted_imu_rotation_delta
                )
                self.publish_status_msg(
                    accepted=False,
                    reason=self.last_recovery_status["reason"],
                    stamp=msg.header.stamp,
                    points=int(current_points.shape[0]),
                    fitness=self.last_recovery_status["fitness"],
                    rmse=self.last_recovery_status["rmse"],
                    processing_ms=self.elapsed_ms(start_time),
                    icp_target="recovery_map",
                    keyframe_added=False,
                    gate=recovery_gate,
                    prediction=prediction,
                )
                return

            # Alternate recovery frame: use exactly ONE scan-to-scan ICP to
            # keep tracking_guess_pose following the robot while the validated
            # map remains untouched.
            prediction = self.update_tracking_prediction(
                cloud=cloud,
                stamp=msg.header.stamp,
                imu_rotation_delta=scan_imu_rotation_delta,
            )
            self.prev_scan_cloud = cloud
            self.prev_scan_stamp = msg.header.stamp
            self.prev_scan_imu_rotation = scan_imu_rotation.copy()

            if prediction["valid"]:
                self.recovery_tracking_valid_streak += 1
            else:
                self.recovery_tracking_valid_streak = 0

            # A single scan-to-scan match is not enough to leave recovery.
            # After several consecutive geometrically valid predictions,
            # validate the accumulated tracking pose with a cumulative
            # motion/yaw gate and promote it back to the public pose.
            if (
                prediction["valid"]
                and self.recovery_tracking_valid_streak
                >= self.recovery_tracking_confirmations_required
            ):
                recovery_gate = self.evaluate_recovery_tracking_candidate(
                    candidate_pose=self.tracking_guess_pose,
                    stamp=msg.header.stamp,
                    imu_rotation_delta=accepted_imu_rotation_delta,
                )

                if recovery_gate["accepted"]:
                    self.pose_matrix = self.tracking_guess_pose.copy()
                    self.last_accepted_stamp = msg.header.stamp
                    self.last_accepted_imu_rotation = (
                        scan_imu_rotation.copy()
                    )
                    self.consecutive_rejections = 0
                    self.recovery_tracking_valid_streak = 0

                    aligned_points = self.transform_points_to_map(
                        current_points
                    )
                    keyframe_added = self.try_add_keyframe(aligned_points)

                    aligned_output = self.apply_transform_to_points(
                        aligned_source_points, self.pose_matrix
                    )
                    self.publish_aligned_cloud(
                        aligned_output, msg.header.stamp
                    )
                    if keyframe_added:
                        self.publish_local_map(msg.header.stamp)
                    self.publish_outputs(msg.header.stamp)

                    self.last_recovery_status.update(
                        {
                            "active": True,
                            "attempted": False,
                            "success": True,
                            "reason": (
                                "recovered_scan_to_scan_confirmed"
                            ),
                            "fitness": prediction["fitness"],
                            "rmse": prediction["rmse"],
                            "map_points": (
                                self.get_recovery_map_point_count()
                            ),
                            "map_source": (
                                self.current_recovery_map_source
                            ),
                        }
                    )

                    self.publish_status_msg(
                        accepted=True,
                        reason="recovered_scan_to_scan_confirmed",
                        stamp=msg.header.stamp,
                        points=int(current_points.shape[0]),
                        fitness=prediction["fitness"],
                        rmse=prediction["rmse"],
                        processing_ms=self.elapsed_ms(start_time),
                        icp_target="previous_cloud_recovery",
                        keyframe_added=keyframe_added,
                        gate=recovery_gate,
                        prediction=prediction,
                    )
                    return

                # The accumulated tracking solution itself is implausible;
                # require a new sequence of valid scan-to-scan matches.
                self.recovery_tracking_valid_streak = 0

            self.consecutive_rejections += 1
            self.last_recovery_status.update(
                {
                    "active": True,
                    "attempted": False,
                    "success": False,
                    "reason": "recovery_tracking_scan_to_scan",
                    "map_points": self.get_recovery_map_point_count(),
                    "map_source": self.current_recovery_map_source,
                }
            )
            self.publish_status_msg(
                accepted=False,
                reason="recovery_tracking_scan_to_scan",
                stamp=msg.header.stamp,
                points=int(current_points.shape[0]),
                fitness=prediction["fitness"],
                rmse=prediction["rmse"],
                processing_ms=self.elapsed_ms(start_time),
                icp_target="previous_cloud_recovery",
                keyframe_added=False,
                gate=self.empty_gate(
                    msg.header.stamp, accepted_imu_rotation_delta
                ),
                prediction=prediction,
            )
            return

        # --------------------------------------------------------
        # FAST PATH: one scan-to-map ICP
        # --------------------------------------------------------

        if use_local_map:
            icp_target = "local_map_fast"

            # Advance only the rotational part of the initial guess with the
            # current IMU increment. Translation remains at the previous
            # tracking estimate; at 10 Hz the map ICP should solve the small
            # inter-scan translation directly.
            initial_guess = self.tracking_guess_pose.copy()
            imu_relative_guess = self.build_relative_initial_guess(
                scan_imu_rotation_delta
            )
            initial_guess = initial_guess @ imu_relative_guess

            if self.planar_mode:
                initial_guess = self.project_transform_to_2d(
                    initial_guess
                )

            result, icp_exception = self.run_icp(
                source_cloud=cloud,
                target_cloud=self.local_map_cloud,
                initial_guess=initial_guess,
                max_correspondence_distance=(
                    self.icp_max_correspondence_distance
                ),
                max_iterations=self.icp_max_iterations,
            )

            primary_reason = "icp_accepted"
            if icp_exception is not None:
                primary_reason = f"icp_exception: {icp_exception}"
            else:
                fitness = float(result.fitness)
                rmse = float(result.inlier_rmse)
                candidate_pose = np.asarray(
                    result.transformation, dtype=np.float64
                )

                if self.planar_mode:
                    candidate_pose = self.project_transform_to_2d(
                        candidate_pose
                    )

                if self.publish_debug_clouds:
                    self.publish_icp_candidate_cloud(
                        debug_source_points,
                        candidate_pose,
                        msg.header.stamp,
                    )

                if fitness < self.min_fitness:
                    primary_reason = "low_fitness"
                else:
                    gate = self.evaluate_pose_candidate(
                        candidate_pose=candidate_pose,
                        stamp=msg.header.stamp,
                        rmse=rmse,
                        imu_rotation_delta=accepted_imu_rotation_delta,
                    )
                    if self.plausibility_enabled and not gate["accepted"]:
                        primary_reason = gate["reason"]

            primary_accepted = (
                icp_exception is None
                and candidate_pose is not None
                and fitness is not None
                and fitness >= self.min_fitness
                and (
                    not self.plausibility_enabled
                    or (gate is not None and gate["accepted"])
                )
            )

            if primary_accepted:
                # Every processed scan remains available as a future fallback
                # scan-to-scan reference, even though no predictor ICP was
                # needed on this healthy frame.
                self.prev_scan_cloud = cloud
                self.prev_scan_stamp = msg.header.stamp
                self.prev_scan_imu_rotation = (
                    scan_imu_rotation.copy()
                )

                self.pose_matrix = candidate_pose
                self.tracking_guess_pose = candidate_pose.copy()
                self.last_accepted_stamp = msg.header.stamp
                self.last_accepted_imu_rotation = (
                    scan_imu_rotation.copy()
                )
                self.consecutive_rejections = 0
                self.recovery_tracking_valid_streak = 0

                aligned_points = self.transform_points_to_map(
                    current_points
                )
                keyframe_added = self.try_add_keyframe(aligned_points)

                aligned_output = self.apply_transform_to_points(
                    aligned_source_points, self.pose_matrix
                )
                self.publish_aligned_cloud(
                    aligned_output, msg.header.stamp
                )

                if self.publish_debug_clouds:
                    self.publish_diagnostic_clouds(
                        debug_source_points, msg.header.stamp
                    )
                    self.publish_deskew_comparison_clouds(
                        raw_no_deskew_display_points,
                        debug_source_points,
                        msg.header.stamp,
                    )

                if keyframe_added:
                    self.publish_local_map(msg.header.stamp)
                self.publish_outputs(msg.header.stamp)

                prediction["reason"] = (
                    "prediction_not_needed_primary_accepted"
                )
                self.publish_status_msg(
                    accepted=True,
                    reason="icp_accepted",
                    stamp=msg.header.stamp,
                    points=int(current_points.shape[0]),
                    fitness=fitness,
                    rmse=rmse,
                    processing_ms=self.elapsed_ms(start_time),
                    icp_target=icp_target,
                    keyframe_added=keyframe_added,
                    gate=gate,
                    prediction=prediction,
                )
                return

            # ----------------------------------------------------
            # PRIMARY REJECTED: defer recovery to the next scan
            # ----------------------------------------------------

            # Do NOT run a second ICP in this same callback. At 10 Hz that
            # was the main source of 250--300 ms spikes exactly when tracking
            # became difficult. Preserve the current scan as the next
            # consecutive reference and advance only IMU rotation cheaply.
            self.prev_scan_cloud = cloud
            self.prev_scan_stamp = msg.header.stamp
            self.prev_scan_imu_rotation = scan_imu_rotation.copy()
            self.advance_tracking_rotation_only(scan_imu_rotation_delta)

            prediction["reason"] = (
                "prediction_deferred_after_primary_rejection"
            )
            self.consecutive_rejections += 1
            rejection_gate = (
                gate
                if gate is not None
                else self.empty_gate(
                    msg.header.stamp,
                    accepted_imu_rotation_delta,
                )
            )
            self.publish_status_msg(
                accepted=False,
                reason=primary_reason,
                stamp=msg.header.stamp,
                points=int(current_points.shape[0]),
                fitness=fitness,
                rmse=rmse,
                processing_ms=self.elapsed_ms(start_time),
                icp_target=icp_target,
                keyframe_added=False,
                gate=rejection_gate,
                prediction=prediction,
            )
            return

        # --------------------------------------------------------
        # BOOTSTRAP: scan-to-scan is still required
        # --------------------------------------------------------

        icp_target = "previous_cloud_bootstrap"
        prediction = self.update_tracking_prediction(
            cloud=cloud,
            stamp=msg.header.stamp,
            imu_rotation_delta=scan_imu_rotation_delta,
        )

        self.prev_scan_cloud = cloud
        self.prev_scan_stamp = msg.header.stamp
        self.prev_scan_imu_rotation = scan_imu_rotation.copy()

        if self.publish_debug_clouds:
            self.publish_diagnostic_clouds(
                debug_source_points, msg.header.stamp
            )
            self.publish_deskew_comparison_clouds(
                raw_no_deskew_display_points,
                debug_source_points,
                msg.header.stamp,
            )

        fitness = prediction["fitness"]
        rmse = prediction["rmse"]

        if not prediction["valid"]:
            self.reject_without_candidate(
                msg.header.stamp,
                start_time,
                prediction["reason"],
                int(current_points.shape[0]),
                fitness=fitness,
                rmse=rmse,
                icp_target=icp_target,
                imu_rotation_delta=accepted_imu_rotation_delta,
                prediction=prediction,
            )
            return

        candidate_pose = self.tracking_guess_pose.copy()

        if self.publish_debug_clouds:
            self.publish_icp_candidate_cloud(
                debug_source_points,
                candidate_pose,
                msg.header.stamp,
            )

        gate = self.evaluate_pose_candidate(
            candidate_pose=candidate_pose,
            stamp=msg.header.stamp,
            rmse=rmse,
            imu_rotation_delta=accepted_imu_rotation_delta,
        )

        if self.plausibility_enabled and not gate["accepted"]:
            self.consecutive_rejections += 1
            self.publish_status_msg(
                accepted=False,
                reason=gate["reason"],
                stamp=msg.header.stamp,
                points=int(current_points.shape[0]),
                fitness=fitness,
                rmse=rmse,
                processing_ms=self.elapsed_ms(start_time),
                icp_target=icp_target,
                keyframe_added=False,
                gate=gate,
                prediction=prediction,
            )
            return

        self.pose_matrix = candidate_pose
        self.tracking_guess_pose = candidate_pose.copy()
        self.last_accepted_stamp = msg.header.stamp
        self.last_accepted_imu_rotation = scan_imu_rotation.copy()
        self.consecutive_rejections = 0
        self.recovery_tracking_valid_streak = 0

        aligned_points = self.transform_points_to_map(current_points)
        keyframe_added = self.try_add_keyframe(aligned_points)

        aligned_output = self.apply_transform_to_points(
            aligned_source_points, self.pose_matrix
        )
        self.publish_aligned_cloud(
            aligned_output, msg.header.stamp
        )

        if keyframe_added:
            self.publish_local_map(msg.header.stamp)
        self.publish_outputs(msg.header.stamp)

        self.publish_status_msg(
            accepted=True,
            reason="icp_accepted",
            stamp=msg.header.stamp,
            points=int(current_points.shape[0]),
            fitness=fitness,
            rmse=rmse,
            processing_ms=self.elapsed_ms(start_time),
            icp_target=icp_target,
            keyframe_added=keyframe_added,
            gate=gate,
            prediction=prediction,
        )

    def update_tracking_prediction(
        self,
        cloud,
        stamp,
        imu_rotation_delta: Optional[np.ndarray],
    ) -> dict:
        """Advance the auxiliary 6-DoF pose with consecutive scan-to-scan ICP.

        The three-axis gyroscope is used only to initialise the relative
        rotation (and as a rotation-only fallback if predictor ICP fails).
        When ICP converges, its COMPLETE 6-DoF transform is preserved:
        X/Y/Z translation plus roll/pitch/yaw rotation.

        No IMU linear acceleration is integrated, so gravity on Z cannot be
        injected into the translational predictor.
        """
        previous_tracking_pose = self.tracking_guess_pose.copy()

        status = self.empty_prediction_status()
        status["attempted"] = bool(self.scan_to_scan_prediction_enabled)
        status["previous_tracking_pose"] = previous_tracking_pose

        relative_guess = self.build_relative_initial_guess(
            imu_rotation_delta
        )

        if not self.scan_to_scan_prediction_enabled:
            relative_motion = relative_guess
            self.tracking_guess_pose = (
                previous_tracking_pose @ relative_motion
            )
            if self.planar_mode:
                self.tracking_guess_pose = self.project_transform_to_2d(
                    self.tracking_guess_pose
                )
            status["reason"] = "prediction_disabled_imu_rotation_only"
            return status

        result, prediction_exception = self.run_icp(
            source_cloud=cloud,
            target_cloud=self.prev_scan_cloud,
            initial_guess=relative_guess,
            max_correspondence_distance=(
                self.prediction_max_correspondence_distance
            ),
            max_iterations=self.prediction_max_iterations,
        )

        if prediction_exception is not None:
            status["reason"] = (
                f"prediction_exception: {prediction_exception}"
            )
            relative_motion = relative_guess
        else:
            fitness = float(result.fitness)
            rmse = float(result.inlier_rmse)
            status["fitness"] = fitness
            status["rmse"] = rmse

            relative_icp = np.asarray(
                result.transformation, dtype=np.float64
            )
            if self.planar_mode:
                relative_icp = self.project_transform_to_2d(relative_icp)

            finite = bool(np.all(np.isfinite(relative_icp)))
            translation_vector = (
                relative_icp[:2, 3]
                if self.planar_mode
                else relative_icp[:3, 3]
            )
            translation_step_m = float(
                np.linalg.norm(translation_vector)
            )
            rotation_step_rad = self.rotation_angle(
                relative_icp[:3, :3]
            )

            dt_scan = 0.0
            if self.prev_scan_stamp is not None:
                dt_scan = max(
                    0.0,
                    self.stamp_to_seconds(stamp)
                    - self.stamp_to_seconds(self.prev_scan_stamp),
                )

            if dt_scan > 1e-6:
                prediction_translation_limit_m = min(
                    self.max_translation_step_m,
                    self.max_linear_speed_mps * dt_scan
                    + self.translation_margin_m,
                )
            else:
                prediction_translation_limit_m = (
                    self.max_translation_step_m
                )

            status["translation_step_m"] = translation_step_m
            status["translation_limit_m"] = float(
                prediction_translation_limit_m
            )
            status["rotation_step_deg"] = math.degrees(rotation_step_rad)

            quality_ok = (
                finite
                and fitness >= self.prediction_min_fitness
                and (
                    self.prediction_max_rmse <= 0.0
                    or rmse <= self.prediction_max_rmse
                )
                and (
                    self.max_translation_step_m <= 0.0
                    or translation_step_m
                    <= prediction_translation_limit_m
                )
            )

            if quality_ok:
                # Crucial 6-DoF change: do NOT overwrite ICP roll/pitch/yaw
                # with IMU yaw. The gyro only supplied the initial guess.
                relative_motion = relative_icp
                status["valid"] = True
                status["reason"] = "prediction_valid_6dof"
            else:
                # Keep tracking orientation alive through short failures using
                # the 3-D gyro delta, but inject no untrusted translation.
                relative_motion = relative_guess

                if not finite:
                    status["reason"] = "prediction_non_finite"
                elif fitness < self.prediction_min_fitness:
                    status["reason"] = "prediction_low_fitness"
                elif (
                    self.prediction_max_rmse > 0.0
                    and rmse > self.prediction_max_rmse
                ):
                    status["reason"] = "prediction_high_rmse"
                else:
                    status["reason"] = (
                        "prediction_implausible_translation"
                    )

        self.tracking_guess_pose = (
            previous_tracking_pose @ relative_motion
        )
        if self.planar_mode:
            self.tracking_guess_pose = self.project_transform_to_2d(
                self.tracking_guess_pose
            )

        return status

    def run_icp(
        self,
        source_cloud,
        target_cloud,
        initial_guess: np.ndarray,
        max_correspondence_distance: float,
        max_iterations: int,
    ):
        icp_start = time.perf_counter()
        self.current_profile_icp_calls += 1
        try:
            result = o3d.pipelines.registration.registration_icp(
                source_cloud,
                target_cloud,
                max_correspondence_distance,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-4,
                    relative_rmse=1e-4,
                    max_iteration=max_iterations,
                ),
            )
            return result, None
        except Exception as exc:
            return None, exc
        finally:
            self.current_profile_icp_ms += (
                time.perf_counter() - icp_start
            ) * 1000.0

    def build_relative_initial_guess(
        self, imu_rotation_delta: Optional[np.ndarray]
    ) -> np.ndarray:
        guess = np.eye(4, dtype=np.float64)

        if (
            self.use_imu_rotation_initial_guess
            and imu_rotation_delta is not None
        ):
            rotation = np.asarray(
                imu_rotation_delta, dtype=np.float64
            ).reshape(3, 3)
            if np.all(np.isfinite(rotation)):
                guess[:3, :3] = rotation

        if self.planar_mode:
            guess = self.project_transform_to_2d(guess)

        return guess

    @staticmethod
    def empty_prediction_status() -> dict:
        return {
            "attempted": False,
            "valid": False,
            "reason": "not_attempted",
            "fitness": None,
            "rmse": None,
            "translation_step_m": None,
            "translation_limit_m": None,
            "rotation_step_deg": None,
            "previous_tracking_pose": None,
        }

    def reject_without_candidate(
        self,
        stamp,
        start_time: float,
        reason: str,
        points: int,
        fitness=None,
        rmse=None,
        icp_target: str = "none",
        imu_rotation_delta: Optional[np.ndarray] = None,
        prediction: Optional[dict] = None,
    ):
        self.consecutive_rejections += 1
        gate = self.empty_gate(stamp, imu_rotation_delta)
        self.publish_status_msg(
            accepted=False,
            reason=reason,
            stamp=stamp,
            points=points,
            fitness=fitness,
            rmse=rmse,
            processing_ms=self.elapsed_ms(start_time),
            icp_target=icp_target,
            keyframe_added=False,
            gate=gate,
            prediction=(
                self.empty_prediction_status()
                if prediction is None
                else prediction
            ),
        )

    # ============================================================
    # PLAUSIBILITY GATE
    # ============================================================

    def evaluate_pose_candidate(
        self,
        candidate_pose: np.ndarray,
        stamp,
        rmse: float,
        imu_rotation_delta: Optional[np.ndarray],
    ) -> dict:
        """Validate a 2-D or 6-DoF pose against the last published pose."""
        dt_sec = self.get_dt_since_last_accepted(stamp)
        relative = np.linalg.inv(self.pose_matrix) @ candidate_pose

        if self.planar_mode:
            relative = self.project_transform_to_2d(relative)

        translation_vector = (
            relative[:2, 3]
            if self.planar_mode
            else relative[:3, 3]
        )
        translation_step_m = float(np.linalg.norm(translation_vector))

        yaw_step_rad = self.normalize_angle(
            self.get_yaw_from_matrix(relative)
        )
        rotation_step_rad = self.rotation_angle(relative[:3, :3])

        if dt_sec > 1e-6:
            translation_limit_m = min(
                self.max_translation_step_m,
                self.max_linear_speed_mps * dt_sec + self.translation_margin_m,
            )
            rotation_limit_rad = min(
                self.max_yaw_step_rad,
                self.max_yaw_rate_rad_s * dt_sec + self.yaw_margin_rad,
            )
        else:
            translation_limit_m = self.max_translation_step_m
            rotation_limit_rad = self.max_yaw_step_rad

        imu_rotation_error_rad = None
        imu_yaw_error_rad = None
        imu_delta_rpy = None
        if imu_rotation_delta is not None:
            imu_rotation_delta = np.asarray(
                imu_rotation_delta, dtype=np.float64
            ).reshape(3, 3)
            if np.all(np.isfinite(imu_rotation_delta)):
                # Keep the complete SO(3) discrepancy as a diagnostic only.
                # The configured gate is explicitly a YAW gate and therefore
                # must not reject normal G1 roll/pitch oscillation.
                rotation_error = (
                    imu_rotation_delta.T @ relative[:3, :3]
                )
                imu_rotation_error_rad = self.rotation_angle(rotation_error)
                imu_delta_rpy = self.rpy_from_rotation(imu_rotation_delta)

                imu_yaw_delta_rad = self.normalize_angle(
                    self.get_yaw_from_rotation(imu_rotation_delta)
                )
                imu_yaw_error_rad = abs(
                    self.normalize_angle(
                        yaw_step_rad - imu_yaw_delta_rad
                    )
                )

        accepted = True
        reason = "plausible"

        if not np.all(np.isfinite(candidate_pose)):
            accepted = False
            reason = "non_finite_pose"
        elif not math.isfinite(rmse):
            accepted = False
            reason = "non_finite_rmse"
        elif self.max_icp_rmse > 0.0 and rmse > self.max_icp_rmse:
            accepted = False
            reason = "high_rmse"
        elif (
            self.max_translation_step_m > 0.0
            and translation_step_m > translation_limit_m
        ):
            accepted = False
            reason = "implausible_translation"
        elif (
            self.max_yaw_step_rad > 0.0
            and rotation_step_rad > rotation_limit_rad
        ):
            accepted = False
            reason = "implausible_rotation"
        elif (
            self.use_imu_yaw_gate
            and imu_yaw_error_rad is not None
            and self.max_imu_yaw_difference_rad > 0.0
            and imu_yaw_error_rad > self.max_imu_yaw_difference_rad
        ):
            accepted = False
            reason = "imu_yaw_mismatch"

        return {
            "accepted": accepted,
            "reason": reason,
            "dt_sec": float(dt_sec),
            "translation_step_m": translation_step_m,
            "yaw_step_rad": float(yaw_step_rad),
            "rotation_step_rad": float(rotation_step_rad),
            # Old yaw fields are retained for CSV/status compatibility. The
            # IMU yaw delta is simply the yaw component of the full 3-D delta.
            "imu_yaw_delta_rad": (
                None
                if imu_delta_rpy is None
                else float(imu_delta_rpy[2])
            ),
            "imu_yaw_error_rad": (
                None
                if imu_yaw_error_rad is None
                else float(imu_yaw_error_rad)
            ),
            "imu_rotation_error_rad": (
                None
                if imu_rotation_error_rad is None
                else float(imu_rotation_error_rad)
            ),
            "imu_rotation_delta_rpy_rad": (
                None
                if imu_delta_rpy is None
                else [float(v) for v in imu_delta_rpy]
            ),
            "translation_limit_m": float(translation_limit_m),
            "yaw_limit_rad": float(rotation_limit_rad),
            "rotation_limit_rad": float(rotation_limit_rad),
        }

    def evaluate_recovery_tracking_candidate(
        self,
        candidate_pose: np.ndarray,
        stamp,
        imu_rotation_delta: Optional[np.ndarray],
    ) -> dict:
        """Validate a scan-to-scan recovery accumulated since last accept.

        Unlike the normal per-frame plausibility gate, the translation limit
        here scales with the complete elapsed time since the last accepted
        pose.  This is necessary because ``tracking_guess_pose`` contains the
        accumulated motion across several rejected scans.
        """
        dt_sec = self.get_dt_since_last_accepted(stamp)
        relative = np.linalg.inv(self.pose_matrix) @ candidate_pose

        if self.planar_mode:
            relative = self.project_transform_to_2d(relative)

        translation_vector = (
            relative[:2, 3]
            if self.planar_mode
            else relative[:3, 3]
        )
        translation_step_m = float(np.linalg.norm(translation_vector))
        yaw_step_rad = self.normalize_angle(
            self.get_yaw_from_matrix(relative)
        )
        rotation_step_rad = self.rotation_angle(relative[:3, :3])

        if dt_sec > 1e-6:
            translation_limit_m = (
                self.max_linear_speed_mps * dt_sec
                + self.translation_margin_m
            )
            rotation_limit_rad = min(
                math.pi,
                self.max_yaw_rate_rad_s * dt_sec
                + self.yaw_margin_rad,
            )
        else:
            translation_limit_m = self.max_translation_step_m
            rotation_limit_rad = self.max_yaw_step_rad

        imu_rotation_error_rad = None
        imu_yaw_error_rad = None
        imu_delta_rpy = None
        if imu_rotation_delta is not None:
            imu_rotation_delta = np.asarray(
                imu_rotation_delta, dtype=np.float64
            ).reshape(3, 3)
            if np.all(np.isfinite(imu_rotation_delta)):
                rotation_error = (
                    imu_rotation_delta.T @ relative[:3, :3]
                )
                imu_rotation_error_rad = self.rotation_angle(
                    rotation_error
                )
                imu_delta_rpy = self.rpy_from_rotation(
                    imu_rotation_delta
                )
                imu_yaw_delta_rad = self.normalize_angle(
                    self.get_yaw_from_rotation(imu_rotation_delta)
                )
                imu_yaw_error_rad = abs(
                    self.normalize_angle(
                        yaw_step_rad - imu_yaw_delta_rad
                    )
                )

        accepted = bool(np.all(np.isfinite(candidate_pose)))
        reason = "recovery_tracking_plausible"

        if not accepted:
            reason = "recovery_tracking_non_finite"
        elif (
            self.max_linear_speed_mps > 0.0
            and translation_step_m > translation_limit_m
        ):
            accepted = False
            reason = "recovery_tracking_translation"
        elif (
            self.max_yaw_rate_rad_s > 0.0
            and rotation_step_rad > rotation_limit_rad
        ):
            accepted = False
            reason = "recovery_tracking_rotation"
        elif (
            self.use_imu_yaw_gate
            and imu_yaw_error_rad is not None
            and self.max_imu_yaw_difference_rad > 0.0
            and imu_yaw_error_rad > self.max_imu_yaw_difference_rad
        ):
            accepted = False
            reason = "recovery_tracking_imu_yaw_mismatch"

        return {
            "accepted": accepted,
            "reason": reason,
            "dt_sec": float(dt_sec),
            "translation_step_m": translation_step_m,
            "yaw_step_rad": float(yaw_step_rad),
            "rotation_step_rad": float(rotation_step_rad),
            "imu_yaw_delta_rad": (
                None
                if imu_delta_rpy is None
                else float(imu_delta_rpy[2])
            ),
            "imu_yaw_error_rad": (
                None
                if imu_yaw_error_rad is None
                else float(imu_yaw_error_rad)
            ),
            "imu_rotation_error_rad": (
                None
                if imu_rotation_error_rad is None
                else float(imu_rotation_error_rad)
            ),
            "imu_rotation_delta_rpy_rad": (
                None
                if imu_delta_rpy is None
                else [float(v) for v in imu_delta_rpy]
            ),
            "translation_limit_m": float(translation_limit_m),
            "yaw_limit_rad": float(rotation_limit_rad),
            "rotation_limit_rad": float(rotation_limit_rad),
        }

    def get_dt_since_last_accepted(self, stamp) -> float:
        if self.last_accepted_stamp is None:
            return 0.0
        return max(
            0.0,
            self.stamp_to_seconds(stamp)
            - self.stamp_to_seconds(self.last_accepted_stamp),
        )

    def empty_gate(
        self, stamp, imu_rotation_delta: Optional[np.ndarray]
    ) -> dict:
        dt_sec = self.get_dt_since_last_accepted(stamp)

        if dt_sec > 1e-6:
            translation_limit_m = min(
                self.max_translation_step_m,
                self.max_linear_speed_mps * dt_sec + self.translation_margin_m,
            )
            rotation_limit_rad = min(
                self.max_yaw_step_rad,
                self.max_yaw_rate_rad_s * dt_sec + self.yaw_margin_rad,
            )
        else:
            translation_limit_m = self.max_translation_step_m
            rotation_limit_rad = self.max_yaw_step_rad

        imu_delta_rpy = None
        if imu_rotation_delta is not None:
            imu_rotation_delta = np.asarray(
                imu_rotation_delta, dtype=np.float64
            ).reshape(3, 3)
            if np.all(np.isfinite(imu_rotation_delta)):
                imu_delta_rpy = self.rpy_from_rotation(imu_rotation_delta)

        return {
            "accepted": False,
            "reason": "no_candidate",
            "dt_sec": float(dt_sec),
            "translation_step_m": None,
            "yaw_step_rad": 0.0,
            "rotation_step_rad": 0.0,
            "imu_yaw_delta_rad": (
                None if imu_delta_rpy is None else float(imu_delta_rpy[2])
            ),
            "imu_yaw_error_rad": None,
            "imu_rotation_error_rad": None,
            "imu_rotation_delta_rpy_rad": (
                None
                if imu_delta_rpy is None
                else [float(v) for v in imu_delta_rpy]
            ),
            "translation_limit_m": float(translation_limit_m),
            "yaw_limit_rad": float(rotation_limit_rad),
            "rotation_limit_rad": float(rotation_limit_rad),
        }

    @staticmethod
    def zero_gate() -> dict:
        return {
            "accepted": True,
            "reason": "initial",
            "dt_sec": 0.0,
            "translation_step_m": 0.0,
            "yaw_step_rad": 0.0,
            "rotation_step_rad": 0.0,
            "imu_yaw_delta_rad": 0.0,
            "imu_yaw_error_rad": 0.0,
            "imu_rotation_error_rad": 0.0,
            "imu_rotation_delta_rpy_rad": [0.0, 0.0, 0.0],
            "translation_limit_m": 0.0,
            "yaw_limit_rad": 0.0,
            "rotation_limit_rad": 0.0,
        }

    # ============================================================
    # RELOCALISATION / RECOVERY
    # ============================================================

    @staticmethod
    def empty_recovery_status() -> dict:
        return {
            "active": False,
            "attempted": False,
            "success": False,
            "reason": "inactive",
            "fitness": None,
            "rmse": None,
            "hypotheses": 0,
            "map_points": 0,
            "map_source": "none",
            "translation_from_tracking_m": None,
            "rotation_from_tracking_deg": None,
        }

    def optimized_map_callback(self, msg: PointCloud2):
        """Cache the latest optimized map message without processing it.

        Conversion/downsampling is deliberately deferred until recovery is
        actually needed so the healthy 10 Hz front-end path stays cheap.
        """
        with self.recovery_map_lock:
            self.latest_optimized_map_msg = msg
            self.optimized_recovery_map_points = None
            self.optimized_recovery_map_cloud = None

    def get_recovery_map_point_count(self) -> int:
        if (
            self.current_recovery_map_source == "optimized_backend"
            and self.optimized_recovery_map_points is not None
        ):
            return int(self.optimized_recovery_map_points.shape[0])
        if self.recovery_map_points is not None:
            return int(self.recovery_map_points.shape[0])
        return int(sum(frame.shape[0] for frame in self.recovery_map_keyframes))

    def get_optimized_recovery_map_cloud(self):
        if not self.recovery_use_optimized_map:
            return None

        with self.recovery_map_lock:
            latest_msg = self.latest_optimized_map_msg
            cached_cloud = self.optimized_recovery_map_cloud

        if latest_msg is None:
            return None
        if cached_cloud is not None:
            return cached_cloud

        try:
            points = self.pointcloud2_to_xyz_array(latest_msg)
            if points.size == 0:
                return None
            points = self.voxel_downsample_points(
                points, self.recovery_map_voxel_size
            )
            if self.recovery_max_map_points > 0:
                points = self.deterministic_limit_points(
                    points, self.recovery_max_map_points
                )
            cloud = self.create_open3d_cloud_from_points(
                points, min_points=1
            )
            if cloud is None:
                return None

            with self.recovery_map_lock:
                # Cache only if no newer optimized map replaced this message.
                if self.latest_optimized_map_msg is latest_msg:
                    self.optimized_recovery_map_points = points
                    self.optimized_recovery_map_cloud = cloud

            return cloud
        except Exception as exc:
            self.get_logger().warn(
                f"Could not prepare optimized recovery map: {exc}"
            )
            return None

    def get_recovery_map_cloud(self):
        """Return optimized backend map when available, else frontend history."""
        optimized = self.get_optimized_recovery_map_cloud()
        if optimized is not None:
            self.current_recovery_map_source = "optimized_backend"
            return optimized

        self.current_recovery_map_source = "historical_frontend"
        if not self.recovery_map_dirty and self.recovery_map_cloud is not None:
            return self.recovery_map_cloud

        frames = list(self.recovery_map_keyframes)
        if not frames:
            self.recovery_map_points = None
            self.recovery_map_cloud = None
            self.recovery_map_dirty = False
            return None

        points = frames[0].copy() if len(frames) == 1 else np.vstack(frames)
        points = self.voxel_downsample_points(
            points, self.recovery_map_voxel_size
        )
        if self.recovery_max_map_points > 0:
            points = self.deterministic_limit_points(
                points, self.recovery_max_map_points
            )

        self.recovery_map_points = points
        self.recovery_map_cloud = self.create_open3d_cloud_from_points(
            points, min_points=1
        )
        self.recovery_map_dirty = False
        return self.recovery_map_cloud

    def advance_tracking_rotation_only(
        self, imu_rotation_delta: Optional[np.ndarray]
    ):
        """Cheaply keep the recovery guess oriented without a second ICP."""
        relative = self.build_relative_initial_guess(imu_rotation_delta)
        self.tracking_guess_pose = self.tracking_guess_pose @ relative
        if self.planar_mode:
            self.tracking_guess_pose = self.project_transform_to_2d(
                self.tracking_guess_pose
            )

    def next_recovery_initial_guess(
        self, tracking_guess: np.ndarray
    ) -> np.ndarray:
        """Return one rotating yaw hypothesis per recovery attempt."""
        offsets = []
        seen = set()
        for value in self.recovery_yaw_offsets_deg:
            offset = float(value)
            key = round(offset, 6)
            if key not in seen:
                seen.add(key)
                offsets.append(offset)

        if not offsets:
            offsets = [0.0]

        offset = offsets[self.recovery_yaw_seed_index % len(offsets)]
        self.recovery_yaw_seed_index += 1

        guess = np.asarray(tracking_guess, dtype=np.float64).copy()
        yaw_offset = self.yaw_to_transform(math.radians(offset))[:3, :3]
        guess[:3, :3] = yaw_offset @ guess[:3, :3]
        if self.planar_mode:
            guess = self.project_transform_to_2d(guess)
        return guess

    def try_relocalize(self, cloud, tracking_guess: np.ndarray) -> dict:
        """Try coarse scan-to-historical-map registration from a few yaw seeds."""
        status = self.empty_recovery_status()
        status["active"] = True
        status["attempted"] = True

        recovery_map_cloud = self.get_recovery_map_cloud()
        status["map_points"] = self.get_recovery_map_point_count()
        status["map_source"] = self.current_recovery_map_source
        if recovery_map_cloud is None:
            status["reason"] = "recovery_map_unavailable"
            return {"pose": None, "status": status}

        source_points = np.asarray(cloud.points)
        if self.recovery_source_voxel_size > self.voxel_size + 1e-9:
            source_points = self.voxel_downsample_points(
                source_points, self.recovery_source_voxel_size
            )
        source_cloud = self.create_open3d_cloud_from_points(
            source_points, min_points=max(50, min(self.min_points, 300))
        )
        if source_cloud is None:
            status["reason"] = "recovery_source_too_sparse"
            return {"pose": None, "status": status}

        best_pose = None
        best_fitness = -1.0
        best_rmse = float("inf")
        best_trans = None
        best_rot = None

        guess = self.next_recovery_initial_guess(tracking_guess)
        status["hypotheses"] = 1

        result, exc = self.run_icp(
            source_cloud=source_cloud,
            target_cloud=recovery_map_cloud,
            initial_guess=guess,
            max_correspondence_distance=(
                self.recovery_max_correspondence_distance
            ),
            max_iterations=self.recovery_max_iterations,
        )

        if exc is None:
            candidate = np.asarray(result.transformation, dtype=np.float64)
            if self.planar_mode:
                candidate = self.project_transform_to_2d(candidate)

            if np.all(np.isfinite(candidate)):
                fitness = float(result.fitness)
                rmse = float(result.inlier_rmse)
                relative_to_tracking = (
                    np.linalg.inv(tracking_guess) @ candidate
                )
                translation = float(
                    np.linalg.norm(relative_to_tracking[:3, 3])
                )
                rotation = self.rotation_angle(
                    relative_to_tracking[:3, :3]
                )

                valid = (
                    fitness >= self.recovery_min_fitness
                    and math.isfinite(rmse)
                    and (
                        self.recovery_max_rmse <= 0.0
                        or rmse <= self.recovery_max_rmse
                    )
                    and (
                        self.recovery_max_translation_from_tracking_m <= 0.0
                        or translation
                        <= self.recovery_max_translation_from_tracking_m
                    )
                    and (
                        self.recovery_max_rotation_from_tracking_rad <= 0.0
                        or rotation
                        <= self.recovery_max_rotation_from_tracking_rad
                    )
                )
                if valid:
                    best_pose = candidate
                    best_fitness = fitness
                    best_rmse = rmse
                    best_trans = translation
                    best_rot = rotation

        if best_pose is None:
            status["reason"] = "relocalization_no_valid_hypothesis"
            return {"pose": None, "status": status}

        status.update(
            {
                "success": True,
                "reason": "relocalized",
                "fitness": float(best_fitness),
                "rmse": float(best_rmse),
                "translation_from_tracking_m": float(best_trans),
                "rotation_from_tracking_deg": math.degrees(best_rot),
            }
        )
        return {"pose": best_pose, "status": status}

    def build_recovery_gate_for_status(
        self, previous_pose: np.ndarray, candidate_pose: np.ndarray, stamp,
        imu_rotation_delta: Optional[np.ndarray],
    ) -> dict:
        gate = self.empty_gate(stamp, imu_rotation_delta)
        relative = np.linalg.inv(previous_pose) @ candidate_pose
        gate["accepted"] = True
        gate["reason"] = "relocalized"
        gate["translation_step_m"] = float(
            np.linalg.norm(relative[:3, 3])
        )
        gate["yaw_step_rad"] = self.normalize_angle(
            self.get_yaw_from_matrix(relative)
        )
        gate["rotation_step_rad"] = self.rotation_angle(
            relative[:3, :3]
        )
        return gate

    def try_add_keyframe(self, aligned_points: np.ndarray) -> bool:
        if not self.enable_local_map or aligned_points.size == 0:
            return False

        if not self.should_add_keyframe():
            return False

        keyframe_points = self.voxel_downsample_points(
            aligned_points.copy(), self.local_map_voxel_size
        )

        if keyframe_points.size == 0:
            return False

        # deque(maxlen=N) removes the oldest keyframe automatically.
        self.local_map_keyframes.append(keyframe_points)
        self.rebuild_local_map_from_recent_keyframes()

        # Keep a longer validated history for relocalisation. Do not build the
        # historical cloud here: mark it dirty and rebuild lazily only if
        # tracking is actually lost.
        recovery_keyframe = self.voxel_downsample_points(
            aligned_points.copy(), self.recovery_map_voxel_size
        )
        if recovery_keyframe.size > 0:
            self.recovery_map_keyframes.append(recovery_keyframe)
            self.recovery_map_dirty = True

        self.local_map_frame_count = len(self.local_map_keyframes)
        self.keyframe_count += 1
        self.last_keyframe_pose = self.pose_matrix.copy()
        return True

    def rebuild_local_map_from_recent_keyframes(self):
        """Rebuild map exclusively from the rolling keyframe deque.

        The point budget is distributed across the retained keyframes so a
        spatial nearest-point selection cannot preserve arbitrary old geometry.
        """
        frames = list(self.local_map_keyframes)

        if not frames:
            self.local_map_points = None
            self.local_map_cloud = None
            return

        if self.max_local_map_points > 0:
            per_frame_budget = max(
                1, self.max_local_map_points // len(frames)
            )
            frames = [
                self.deterministic_limit_points(frame, per_frame_budget)
                for frame in frames
            ]

        rolling_points = (
            frames[0].copy() if len(frames) == 1 else np.vstack(frames)
        )
        rolling_points = self.voxel_downsample_points(
            rolling_points, self.local_map_voxel_size
        )

        if self.max_local_map_points > 0:
            rolling_points = self.deterministic_limit_points(
                rolling_points, self.max_local_map_points
            )

        self.local_map_points = rolling_points
        self.local_map_cloud = self.create_open3d_cloud_from_points(
            rolling_points, min_points=1
        )

    def should_add_keyframe(self) -> bool:
        """Select keyframes from horizontal motion only.

        Registration remains full 6-DoF, but the Unitree G1's walking motion
        introduces real roll/pitch oscillation and small vertical motion.
        Those components must not create extra keyframes. A keyframe is
        therefore triggered only by XY displacement or yaw change.
        """
        if self.last_keyframe_pose is None:
            return True

        # Use absolute map poses for XY distance. This deliberately ignores Z.
        current_xy = self.pose_matrix[:2, 3]
        last_xy = self.last_keyframe_pose[:2, 3]
        translation_delta = float(np.linalg.norm(current_xy - last_xy))

        # Use only heading change. Roll/pitch oscillation from biped walking
        # does not count as motion for local-map keyframe selection.
        current_yaw = self.get_yaw_from_matrix(self.pose_matrix)
        last_yaw = self.get_yaw_from_matrix(self.last_keyframe_pose)
        rotation_delta = abs(
            self.normalize_angle(current_yaw - last_yaw)
        )

        return (
            translation_delta >= self.keyframe_min_translation
            or rotation_delta >= self.keyframe_min_yaw
        )

    def reset_slam_state(self, stamp=None):
        self.reset_counter += 1

        if self.reset_pose_to_identity:
            self.pose_matrix = np.eye(4, dtype=np.float64)

        self.tracking_guess_pose = self.pose_matrix.copy()

        self.prev_scan_cloud = None
        self.prev_scan_stamp = None
        self.prev_scan_imu_rotation = None
        self.last_accepted_stamp = None
        self.last_accepted_imu_rotation = None

        self.cloud_counter = 0
        self.consecutive_rejections = 0
        self.recovery_tracking_valid_streak = 0
        self.recovery_yaw_seed_index = 0
        self.last_processed_lidar_stamp_sec = None
        self.estimated_dropped_scans = 0
        self.pending_lidar_queue_drops = 0

        with self.pending_lidar_lock:
            self.pending_lidar_scans.clear()

        with self.imu_state_lock:
            self.imu_rotation_matrix = np.eye(3, dtype=np.float64)
            self.imu_yaw = 0.0
            self.last_imu_time = None
            self.imu_gyro_buffer.clear()
            self.imu_orientation_buffer.clear()

        self.last_deskew_status = self.empty_deskew_status()

        if self.clear_local_map_on_reset:
            self.local_map_keyframes.clear()
            self.local_map_points = None
            self.local_map_cloud = None
            self.local_map_frame_count = 0
            self.keyframe_count = 0
            self.last_keyframe_pose = None
            self.recovery_map_keyframes.clear()
            self.recovery_map_points = None
            self.recovery_map_cloud = None
            self.recovery_map_dirty = True
            self.latest_optimized_map_msg = None
            self.optimized_recovery_map_points = None
            self.optimized_recovery_map_cloud = None
            self.current_recovery_map_source = "historical_frontend"
            self.last_recovery_status = self.empty_recovery_status()

        if self.clear_path_on_reset:
            self.path_msg = PathMsg()
            self.path_msg.header.frame_id = self.fixed_frame
            if stamp is not None:
                self.path_msg.header.stamp = stamp

        if stamp is not None:
            self.publish_reset_outputs(stamp)

    def publish_reset_outputs(self, stamp):
        if self.clear_path_on_reset:
            self.path_pub.publish(self.path_msg)

        pose_msg = self.build_pose_msg(stamp)
        self.pose_pub.publish(pose_msg)
        self.odom_pub.publish(self.build_odom_msg(stamp, pose_msg))
        self.marker_pub.publish(self.build_marker_msg(stamp, pose_msg))

        if self.publish_tf:
            self.publish_dynamic_tf(stamp)

        empty_cloud = self.create_empty_pointcloud2(stamp, self.fixed_frame)
        self.aligned_cloud_pub.publish(empty_cloud)
        if self.publish_debug_clouds:
            self.raw_aligned_cloud_pub.publish(empty_cloud)
            self.predicted_cloud_pub.publish(empty_cloud)
            self.raw_no_deskew_cloud_pub.publish(empty_cloud)
            self.deskewed_cloud_pub.publish(empty_cloud)
            self.icp_candidate_cloud_pub.publish(empty_cloud)
        self.local_map_pub.publish(empty_cloud)

        self.publish_status_msg(
            accepted=True,
            reason="slam_reset",
            stamp=stamp,
            points=0,
            fitness=None,
            rmse=None,
            processing_ms=0.0,
            icp_target="reset",
            keyframe_added=False,
            gate=self.zero_gate(),
            prediction=self.empty_prediction_status(),
        )

    # ============================================================
    # POINT CLOUD HELPERS
    # ============================================================

    @staticmethod
    def pointcloud2_to_xyz_array(msg):
        """Fast XYZ-only conversion used when per-point timestamps are not needed."""
        try:
            points = point_cloud2.read_points_numpy(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            if points.dtype.names is not None:
                points = np.column_stack(
                    (points["x"], points["y"], points["z"])
                )
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        except (AttributeError, AssertionError, TypeError, ValueError):
            raw_points = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            if (
                isinstance(raw_points, np.ndarray)
                and raw_points.dtype.names is not None
            ):
                points = np.column_stack(
                    (
                        raw_points["x"],
                        raw_points["y"],
                        raw_points["z"],
                    )
                ).astype(np.float64, copy=False)
            else:
                points = np.asarray(
                    list(raw_points), dtype=np.float64
                ).reshape(-1, 3)

        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        finite = np.all(np.isfinite(points), axis=1)
        return points[finite]

    @staticmethod
    def pointcloud2_to_xyz_timestamp_array(msg):
        """Return XYZ points plus the Livox per-point timestamp field.

        For the MID-360 bags used in this project, ``timestamp`` is FLOAT64
        absolute nanoseconds and ``header.stamp`` matches the first point.
        If the timestamp field is unavailable, XYZ is still returned and
        deskew can safely fall back to the original cloud.
        """
        field_names = {field.name for field in msg.fields}
        has_timestamp = "timestamp" in field_names

        if has_timestamp:
            try:
                raw = point_cloud2.read_points(
                    msg,
                    field_names=("x", "y", "z", "timestamp"),
                    skip_nans=True,
                )

                if (
                    isinstance(raw, np.ndarray)
                    and raw.dtype.names is not None
                ):
                    points = np.column_stack(
                        (
                            raw["x"],
                            raw["y"],
                            raw["z"],
                        )
                    ).astype(np.float64, copy=False)
                    timestamps = np.asarray(
                        raw["timestamp"],
                        dtype=np.float64,
                    ).reshape(-1)
                else:
                    rows = list(raw)
                    if not rows:
                        return (
                            np.empty((0, 3), dtype=np.float64),
                            np.empty((0,), dtype=np.float64),
                        )
                    points = np.asarray(
                        [[r[0], r[1], r[2]] for r in rows],
                        dtype=np.float64,
                    )
                    timestamps = np.asarray(
                        [r[3] for r in rows],
                        dtype=np.float64,
                    )

                finite = (
                    np.all(np.isfinite(points), axis=1)
                    & np.isfinite(timestamps)
                )
                return points[finite], timestamps[finite]

            except (AttributeError, AssertionError, TypeError, ValueError):
                pass

        # Timestamp unavailable or heterogeneous read failed: XYZ fallback.
        try:
            points = point_cloud2.read_points_numpy(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            if points.dtype.names is not None:
                points = np.column_stack(
                    (points["x"], points["y"], points["z"])
                )
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        except (AttributeError, AssertionError, TypeError, ValueError):
            raw_points = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            if (
                isinstance(raw_points, np.ndarray)
                and raw_points.dtype.names is not None
            ):
                points = np.column_stack(
                    (
                        raw_points["x"],
                        raw_points["y"],
                        raw_points["z"],
                    )
                ).astype(np.float64, copy=False)
            else:
                points = np.asarray(
                    list(raw_points),
                    dtype=np.float64,
                ).reshape(-1, 3)

        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64), None

        finite = np.all(np.isfinite(points), axis=1)
        return points[finite], None

    @staticmethod
    def empty_deskew_status() -> dict:
        return {
            "enabled": False,
            "applied": False,
            "reason": "not_attempted",
            "timestamp_available": False,
            "scan_duration_ms": None,
            "rotation_correction_deg": None,
            "roll_correction_deg": None,
            "pitch_correction_deg": None,
            "yaw_correction_deg": None,
            "imu_samples": 0,
        }

    def deskew_points_3d(
        self,
        points: np.ndarray,
        point_timestamps_ns,
        scan_stamp,
    ):
        """Rotationally deskew a scan in roll, pitch and yaw.

        Points are brought to the sensor orientation at ``header.stamp``.
        The method integrates the corrected three-axis gyroscope over the
        ~100 ms MID-360 scan. Translation during the scan is intentionally
        not compensated here; X/Y/Z motion is estimated by ICP.
        """
        status = self.empty_deskew_status()
        status["enabled"] = bool(
            self.deskew_enabled and self.deskew_use_imu_rotation
        )
        status["timestamp_available"] = (
            point_timestamps_ns is not None
        )

        if not status["enabled"]:
            status["reason"] = "deskew_disabled"
            return points, status

        if point_timestamps_ns is None or len(point_timestamps_ns) != len(points):
            status["reason"] = "timestamp_unavailable"
            return points, status

        if len(points) == 0:
            status["reason"] = "empty_cloud"
            return points, status

        header_ns = (
            float(scan_stamp.sec) * 1e9
            + float(scan_stamp.nanosec)
        )
        relative_times = (
            np.asarray(point_timestamps_ns, dtype=np.float64)
            - header_ns
        ) * 1e-9

        if not np.all(np.isfinite(relative_times)):
            status["reason"] = "non_finite_point_timestamps"
            return points, status

        min_rel = float(np.min(relative_times))
        max_rel = float(np.max(relative_times))
        scan_duration = max_rel - min_rel
        status["scan_duration_ms"] = scan_duration * 1000.0

        if min_rel < -0.005:
            status["reason"] = "point_timestamp_before_header"
            return points, status

        if (
            self.deskew_max_scan_duration_sec > 0.0
            and scan_duration > self.deskew_max_scan_duration_sec
        ):
            status["reason"] = "scan_duration_too_long"
            return points, status

        with self.imu_state_lock:
            imu_snapshot = list(self.imu_gyro_buffer)

        if len(imu_snapshot) < 2:
            status["reason"] = "insufficient_imu_samples"
            return points, status

        imu = np.asarray(imu_snapshot, dtype=np.float64)
        imu_times = imu[:, 0]
        imu_omega = imu[:, 1:4]

        keep = np.ones(len(imu_times), dtype=bool)
        if len(imu_times) > 1:
            keep[1:] = np.diff(imu_times) > 0.0
        imu_times = imu_times[keep]
        imu_omega = imu_omega[keep]

        if len(imu_times) < 2:
            status["reason"] = "insufficient_unique_imu_samples"
            return points, status

        scan_t0 = self.stamp_to_seconds(scan_stamp)
        scan_t1 = scan_t0 + max_rel

        before_gap = max(0.0, float(imu_times[0] - scan_t0))
        after_gap = max(0.0, float(scan_t1 - imu_times[-1]))
        if (
            before_gap > self.deskew_max_imu_extrapolation_sec
            or after_gap > self.deskew_max_imu_extrapolation_sec
        ):
            status["reason"] = "insufficient_imu_coverage"
            return points, status

        relevant = (
            (imu_times >= scan_t0 - self.deskew_max_imu_extrapolation_sec)
            & (imu_times <= scan_t1 + self.deskew_max_imu_extrapolation_sec)
        )
        rel_imu_times = imu_times[relevant]

        if len(rel_imu_times) < 2:
            status["reason"] = "insufficient_imu_coverage"
            return points, status

        if (
            self.deskew_max_imu_gap_sec > 0.0
            and np.max(np.diff(rel_imu_times))
            > self.deskew_max_imu_gap_sec
        ):
            status["reason"] = "imu_gap_too_large"
            return points, status

        status["imu_samples"] = int(len(rel_imu_times))

        # Knots at scan start/end plus every real IMU sample inside the scan.
        inside = imu_times[(imu_times > scan_t0) & (imu_times < scan_t1)]
        knot_times = np.concatenate(([scan_t0], inside, [scan_t1]))
        knot_times = np.unique(knot_times)

        knot_omega = np.column_stack(
            [
                np.interp(knot_times, imu_times, imu_omega[:, axis])
                for axis in range(3)
            ]
        )

        # Cumulative orientation change from scan start to each knot.
        cumulative_rotations = [np.eye(3, dtype=np.float64)]
        for i in range(len(knot_times) - 1):
            dt = float(knot_times[i + 1] - knot_times[i])
            avg_omega = 0.5 * (knot_omega[i] + knot_omega[i + 1])
            d_rotation = self.rotation_vector_to_matrix(avg_omega * dt)
            cumulative_rotations.append(
                cumulative_rotations[-1] @ d_rotation
            )

        point_times = scan_t0 + relative_times
        interval_idx = np.searchsorted(
            knot_times, point_times, side="right"
        ) - 1
        interval_idx = np.clip(interval_idx, 0, len(knot_times) - 2)

        corrected = np.empty_like(points)

        # Only a few dozen IMU intervals exist per scan. Loop over intervals,
        # not over ~20k LiDAR points.
        for i in np.unique(interval_idx):
            mask = interval_idx == i
            t_left = knot_times[i]
            t_right = knot_times[i + 1]
            w_left = knot_omega[i]
            w_right = knot_omega[i + 1]
            interval = max(float(t_right - t_left), 1e-12)

            dq = point_times[mask] - t_left
            alpha = np.clip(dq / interval, 0.0, 1.0)
            w_query = (
                w_left[None, :]
                + alpha[:, None] * (w_right - w_left)[None, :]
            )

            # Trapezoidal integral of angular rate within this interval.
            partial_rotvec = (
                0.5 * (w_left[None, :] + w_query) * dq[:, None]
            )

            partial_points = self.rotate_points_by_rotation_vectors(
                points[mask], partial_rotvec
            )
            corrected[mask] = (
                partial_points @ cumulative_rotations[i].T
            )

        end_rotation = cumulative_rotations[-1]
        roll, pitch, yaw = self.rpy_from_rotation(end_rotation)
        status["rotation_correction_deg"] = math.degrees(
            self.rotation_angle(end_rotation)
        )
        status["roll_correction_deg"] = math.degrees(roll)
        status["pitch_correction_deg"] = math.degrees(pitch)
        status["yaw_correction_deg"] = math.degrees(yaw)
        status["applied"] = True
        status["reason"] = "deskew_3d_applied"

        return corrected, status

    def range_filter_mask(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0,), dtype=bool)

        distances = np.linalg.norm(points, axis=1)
        return (
            np.isfinite(points).all(axis=1)
            & np.isfinite(distances)
            & (distances >= self.min_range)
            & (distances <= self.max_range)
        )

    def filter_points_by_range(self, points: np.ndarray) -> np.ndarray:
        return points[self.range_filter_mask(points)]

    @staticmethod
    def deterministic_limit_points(
        points: np.ndarray, max_points: int
    ) -> np.ndarray:
        if max_points <= 0 or points.shape[0] <= max_points:
            return points

        indices = np.linspace(
            0,
            points.shape[0] - 1,
            num=max_points,
            dtype=np.int64,
        )
        return points[indices]

    def create_open3d_cloud(self, points: np.ndarray):
        if points.shape[0] < self.min_points:
            return None

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        if self.voxel_size > 0.0:
            cloud = cloud.voxel_down_sample(self.voxel_size)

        if np.asarray(cloud.points).shape[0] < self.min_points:
            return None

        return cloud

    @staticmethod
    def create_open3d_cloud_from_points(points: np.ndarray, min_points: int):
        if points is None or points.shape[0] < min_points:
            return None

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud

    @staticmethod
    def voxel_downsample_points(
        points: np.ndarray, voxel_size: float
    ) -> np.ndarray:
        if points.size == 0 or voxel_size <= 0.0:
            return points

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(voxel_size)
        return np.asarray(cloud.points)

    def transform_points_to_map(self, points: np.ndarray) -> np.ndarray:
        return self.apply_transform_to_points(points, self.pose_matrix)

    # ============================================================
    # PUBLISHING
    # ============================================================

    def publish_aligned_cloud(self, points: np.ndarray, stamp):
        if not self.enable_aligned_cloud or points is None or points.size == 0:
            return
        self.aligned_cloud_pub.publish(
            self.create_pointcloud2_xyz32(points, stamp, self.fixed_frame)
        )

    def publish_diagnostic_clouds(
        self, points_sensor_frame: np.ndarray, stamp
    ):
        """Publish current-scan diagnostics without affecting SLAM state.

        Both outputs contain the same filtered, pre-voxel LiDAR scan. The only
        difference is the pose used to place it in ``map``:

          raw_aligned_cloud -> last validated/public pose (pose_matrix)
          predicted_cloud   -> auxiliary tracking pose (tracking_guess_pose)

        They are intentionally separate from /g1/slam/aligned_cloud because
        the latter is consumed by the global pose-graph backend and therefore
        remains accepted-pose-only.
        """
        if (
            not self.publish_debug_clouds
            or not self.enable_aligned_cloud
            or points_sensor_frame is None
            or points_sensor_frame.size == 0
        ):
            return

        raw_aligned = self.apply_transform_to_points(
            points_sensor_frame, self.pose_matrix
        )
        predicted = self.apply_transform_to_points(
            points_sensor_frame, self.tracking_guess_pose
        )

        self.raw_aligned_cloud_pub.publish(
            self.create_pointcloud2_xyz32(
                raw_aligned, stamp, self.fixed_frame
            )
        )
        self.predicted_cloud_pub.publish(
            self.create_pointcloud2_xyz32(
                predicted, stamp, self.fixed_frame
            )
        )

    def publish_deskew_comparison_clouds(
        self,
        raw_no_deskew_points: np.ndarray,
        deskewed_points: np.ndarray,
        stamp,
    ):
        """Publish raw-vs-deskewed scans with the SAME validated pose.

        These topics isolate the effect of deskew itself. If they separate
        strongly during a turn, the difference is caused by deskew, not by
        the tracking predictor or primary ICP pose.
        """
        if not self.publish_debug_clouds or not self.enable_aligned_cloud:
            return

        if (
            raw_no_deskew_points is not None
            and raw_no_deskew_points.size > 0
        ):
            raw_map = self.apply_transform_to_points(
                raw_no_deskew_points, self.pose_matrix
            )
            self.raw_no_deskew_cloud_pub.publish(
                self.create_pointcloud2_xyz32(
                    raw_map, stamp, self.fixed_frame
                )
            )

        if deskewed_points is not None and deskewed_points.size > 0:
            deskewed_map = self.apply_transform_to_points(
                deskewed_points, self.pose_matrix
            )
            self.deskewed_cloud_pub.publish(
                self.create_pointcloud2_xyz32(
                    deskewed_map, stamp, self.fixed_frame
                )
            )

    def publish_icp_candidate_cloud(
        self,
        points_sensor_frame: np.ndarray,
        candidate_pose: np.ndarray,
        stamp,
    ):
        """Publish the current primary ICP pose hypothesis for diagnostics."""
        if (
            not self.publish_debug_clouds
            or not self.enable_aligned_cloud
            or points_sensor_frame is None
            or points_sensor_frame.size == 0
            or candidate_pose is None
            or not np.all(np.isfinite(candidate_pose))
        ):
            return

        candidate_points = self.apply_transform_to_points(
            points_sensor_frame, candidate_pose
        )
        self.icp_candidate_cloud_pub.publish(
            self.create_pointcloud2_xyz32(
                candidate_points, stamp, self.fixed_frame
            )
        )

    def publish_local_map(self, stamp):
        if (
            not self.enable_local_map
            or self.local_map_points is None
            or self.local_map_points.size == 0
        ):
            return

        # Each publication is a complete replacement snapshot of the current
        # rolling local map. No historical points are appended here.
        self.local_map_pub.publish(
            self.create_pointcloud2_xyz32(
                self.local_map_points, stamp, self.fixed_frame
            )
        )

    @staticmethod
    def create_pointcloud2_xyz32(
        points: np.ndarray, stamp, frame_id: str
    ) -> PointCloud2:
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return point_cloud2.create_cloud_xyz32(
            header, points.astype(np.float32)
        )

    @staticmethod
    def create_empty_pointcloud2(stamp, frame_id: str) -> PointCloud2:
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return point_cloud2.create_cloud_xyz32(
            header, np.empty((0, 3), dtype=np.float32)
        )

    def publish_outputs(self, stamp):
        pose_msg = self.build_pose_msg(stamp)
        self.pose_pub.publish(pose_msg)
        self.odom_pub.publish(self.build_odom_msg(stamp, pose_msg))
        self.marker_pub.publish(self.build_marker_msg(stamp, pose_msg))

        self.path_msg.header.stamp = stamp
        self.path_msg.header.frame_id = self.fixed_frame
        self.path_msg.poses.append(pose_msg)

        if self.max_path_length > 0 and len(self.path_msg.poses) > self.max_path_length:
            self.path_msg.poses = self.path_msg.poses[-self.max_path_length :]

        self.path_pub.publish(self.path_msg)

        if self.publish_tf:
            self.publish_dynamic_tf(stamp)

    def build_pose_msg(self, stamp) -> PoseStamped:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.fixed_frame
        pose_msg.pose.position.x = float(self.pose_matrix[0, 3])
        pose_msg.pose.position.y = float(self.pose_matrix[1, 3])
        pose_msg.pose.position.z = float(self.pose_matrix[2, 3])

        qx, qy, qz, qw = self.quaternion_from_matrix(self.pose_matrix)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        return pose_msg

    def build_odom_msg(self, stamp, pose_msg: PoseStamped) -> Odometry:
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.fixed_frame
        odom_msg.child_frame_id = self.output_frame
        odom_msg.pose.pose = pose_msg.pose
        return odom_msg

    def build_marker_msg(self, stamp, pose_msg: PoseStamped) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.fixed_frame
        marker.ns = "g1_slam"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.0
        marker.color.a = 1.0
        return marker

    def publish_dynamic_tf(self, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.fixed_frame
        transform.child_frame_id = self.output_frame
        transform.transform.translation.x = float(self.pose_matrix[0, 3])
        transform.transform.translation.y = float(self.pose_matrix[1, 3])
        transform.transform.translation.z = float(self.pose_matrix[2, 3])

        qx, qy, qz, qw = self.quaternion_from_matrix(self.pose_matrix)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    def publish_status_msg(
        self,
        accepted: bool,
        reason: str,
        stamp,
        points: int,
        fitness,
        rmse,
        processing_ms: float,
        icp_target: str,
        keyframe_added: bool,
        gate: dict,
        prediction: Optional[dict] = None,
    ):
        if not self.publish_status:
            return

        if prediction is None:
            prediction = self.empty_prediction_status()

        status = {
            "algorithm": "icp",
            "accepted": bool(accepted),
            "reason": reason,
            "stamp_sec": int(stamp.sec),
            "stamp_nanosec": int(stamp.nanosec),
            "points": int(points),
            "fitness": fitness,
            "rmse": rmse,
            "processing_ms": float(processing_ms),
            "profile_preprocess_ms": float(
                self.current_profile_preprocess_ms
            ),
            "profile_deskew_ms": float(
                self.current_profile_deskew_ms
            ),
            "profile_icp_ms": float(
                self.current_profile_icp_ms
            ),
            "profile_icp_calls": int(
                self.current_profile_icp_calls
            ),
            "estimated_dropped_scans": int(
                self.estimated_dropped_scans
            ),
            "pending_lidar_queue_drops": int(
                self.pending_lidar_queue_drops
            ),
            "icp_target": icp_target,
            "keyframe_added": bool(keyframe_added),
            "keyframe_count": int(self.keyframe_count),
            "local_map_frames": int(self.local_map_frame_count),
            "local_map_max_keyframes": int(self.max_local_map_keyframes),
            "local_map_points": int(
                0
                if self.local_map_points is None
                else self.local_map_points.shape[0]
            ),
            "local_map_policy": "rolling_recent_keyframes",
            "position": {
                "x": float(self.pose_matrix[0, 3]),
                "y": float(self.pose_matrix[1, 3]),
                "z": float(self.pose_matrix[2, 3]),
            },
            "yaw_rad": float(self.get_yaw_from_matrix(self.pose_matrix)),
            "rpy_deg": [
                math.degrees(v)
                for v in self.rpy_from_rotation(self.pose_matrix[:3, :3])
            ],
            "planar_mode": bool(self.planar_mode),
            "output_frame": self.output_frame,
            "plausibility_enabled": bool(self.plausibility_enabled),
            "dt_sec": gate["dt_sec"],
            "translation_step_m": gate["translation_step_m"],
            "yaw_step_deg": math.degrees(gate["yaw_step_rad"]),
            "rotation_step_deg": math.degrees(
                gate.get("rotation_step_rad", abs(gate["yaw_step_rad"]))
            ),
            "imu_yaw_delta_deg": (
                None
                if gate["imu_yaw_delta_rad"] is None
                else math.degrees(gate["imu_yaw_delta_rad"])
            ),
            "imu_yaw_error_deg": (
                None
                if gate["imu_yaw_error_rad"] is None
                else math.degrees(gate["imu_yaw_error_rad"])
            ),
            "translation_limit_m": gate["translation_limit_m"],
            "yaw_limit_deg": math.degrees(gate["yaw_limit_rad"]),
            "rotation_limit_deg": math.degrees(
                gate.get("rotation_limit_rad", gate["yaw_limit_rad"])
            ),
            "imu_rotation_error_deg": (
                None
                if gate.get("imu_rotation_error_rad") is None
                else math.degrees(gate["imu_rotation_error_rad"])
            ),
            "imu_rotation_delta_rpy_deg": (
                None
                if gate.get("imu_rotation_delta_rpy_rad") is None
                else [
                    math.degrees(v)
                    for v in gate["imu_rotation_delta_rpy_rad"]
                ]
            ),
            "consecutive_rejections": int(self.consecutive_rejections),
            "prediction_attempted": bool(prediction["attempted"]),
            "prediction_valid": bool(prediction["valid"]),
            "prediction_reason": prediction["reason"],
            "prediction_fitness": prediction["fitness"],
            "prediction_rmse": prediction["rmse"],
            "prediction_translation_step_m": prediction[
                "translation_step_m"
            ],
            "prediction_translation_limit_m": prediction[
                "translation_limit_m"
            ],
            "prediction_rotation_step_deg": prediction.get(
                "rotation_step_deg"
            ),
            "tracking_guess_position": {
                "x": float(self.tracking_guess_pose[0, 3]),
                "y": float(self.tracking_guess_pose[1, 3]),
                "z": float(self.tracking_guess_pose[2, 3]),
            },
            "tracking_guess_yaw_rad": float(
                self.get_yaw_from_matrix(self.tracking_guess_pose)
            ),
            "tracking_guess_rpy_deg": [
                math.degrees(v)
                for v in self.rpy_from_rotation(
                    self.tracking_guess_pose[:3, :3]
                )
            ],
            "deskew_enabled": bool(
                self.last_deskew_status["enabled"]
            ),
            "deskew_applied": bool(
                self.last_deskew_status["applied"]
            ),
            "deskew_reason": self.last_deskew_status["reason"],
            "deskew_timestamp_available": bool(
                self.last_deskew_status["timestamp_available"]
            ),
            "deskew_duration_ms": self.last_deskew_status[
                "scan_duration_ms"
            ],
            "deskew_rotation_deg": self.last_deskew_status.get(
                "rotation_correction_deg"
            ),
            "deskew_roll_deg": self.last_deskew_status.get(
                "roll_correction_deg"
            ),
            "deskew_pitch_deg": self.last_deskew_status.get(
                "pitch_correction_deg"
            ),
            "deskew_yaw_deg": self.last_deskew_status[
                "yaw_correction_deg"
            ],
            "deskew_imu_samples": int(
                self.last_deskew_status["imu_samples"]
            ),
            "recovery_active": bool(
                self.last_recovery_status["active"]
            ),
            "recovery_attempted": bool(
                self.last_recovery_status["attempted"]
            ),
            "recovery_success": bool(
                self.last_recovery_status["success"]
            ),
            "recovery_reason": self.last_recovery_status["reason"],
            "recovery_fitness": self.last_recovery_status["fitness"],
            "recovery_rmse": self.last_recovery_status["rmse"],
            "recovery_hypotheses": int(
                self.last_recovery_status["hypotheses"]
            ),
            "recovery_map_points": int(
                self.last_recovery_status["map_points"]
            ),
            "recovery_map_source": self.last_recovery_status["map_source"],
            "recovery_translation_from_tracking_m": (
                self.last_recovery_status[
                    "translation_from_tracking_m"
                ]
            ),
            "recovery_rotation_from_tracking_deg": (
                self.last_recovery_status[
                    "rotation_from_tracking_deg"
                ]
            ),
        }

        msg = String()
        msg.data = json.dumps(status, separators=(",", ":"))
        self.status_pub.publish(msg)

    # ============================================================
    # TRANSFORMS / MATH
    # ============================================================

    @staticmethod
    def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
        """SO(3) exponential map using Rodrigues' formula."""
        v = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
        theta = float(np.linalg.norm(v))
        if theta < 1e-12:
            # First-order approximation is sufficient near zero.
            x, y, z = v
            return np.array(
                [
                    [1.0, -z, y],
                    [z, 1.0, -x],
                    [-y, x, 1.0],
                ],
                dtype=np.float64,
            )

        axis = v / theta
        x, y, z = axis
        k = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        s = math.sin(theta)
        c = math.cos(theta)
        return np.eye(3, dtype=np.float64) + s * k + (1.0 - c) * (k @ k)

    @staticmethod
    def rotate_points_by_rotation_vectors(
        points: np.ndarray, rotation_vectors: np.ndarray
    ) -> np.ndarray:
        """Rotate N points by N Rodrigues rotation vectors, vectorised."""
        points = np.asarray(points, dtype=np.float64)
        rotvec = np.asarray(rotation_vectors, dtype=np.float64)
        if points.size == 0:
            return points.copy()

        theta = np.linalg.norm(rotvec, axis=1)
        result = points.copy()
        moving = theta > 1e-12
        if not np.any(moving):
            return result

        th = theta[moving]
        k = rotvec[moving] / th[:, None]
        p = points[moving]
        c = np.cos(th)[:, None]
        s = np.sin(th)[:, None]
        cross = np.cross(k, p)
        dot = np.sum(k * p, axis=1)[:, None]
        result[moving] = (
            p * c
            + cross * s
            + k * dot * (1.0 - c)
        )
        return result

    @staticmethod
    def rotation_angle(rotation: np.ndarray) -> float:
        """Return the unsigned SO(3) rotation angle in radians."""
        r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        value = (float(np.trace(r)) - 1.0) * 0.5
        value = max(-1.0, min(1.0, value))
        return math.acos(value)

    @staticmethod
    def rpy_from_rotation(rotation: np.ndarray):
        """Return roll, pitch, yaw (XYZ intrinsic / ROS-style RzRyRx)."""
        r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
        cp = math.cos(pitch)
        if abs(cp) > 1e-8:
            roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
            yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        else:
            roll = 0.0
            yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
        return roll, pitch, yaw

    @staticmethod
    def get_yaw_from_rotation(rotation: np.ndarray) -> float:
        r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        return math.atan2(float(r[1, 0]), float(r[0, 0]))

    @staticmethod
    def build_transform_from_xyz_rpy(
        translation: np.ndarray, rpy: np.ndarray
    ) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float64,
        )
        rot_y = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float64,
        )
        rot_z = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot_z @ rot_y @ rot_x
        transform[:3, 3] = translation
        return transform

    @staticmethod
    def apply_transform_to_points(
        points: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        if points.size == 0:
            return points
        return points @ transform[:3, :3].T + transform[:3, 3]

    def project_transform_to_2d(self, transform: np.ndarray) -> np.ndarray:
        projected = np.eye(4, dtype=np.float64)
        x = float(transform[0, 3])
        y = float(transform[1, 3])
        yaw = self.get_yaw_from_matrix(transform)
        c, s = math.cos(yaw), math.sin(yaw)
        projected[0, 0] = c
        projected[0, 1] = -s
        projected[1, 0] = s
        projected[1, 1] = c
        projected[0, 3] = x
        projected[1, 3] = y
        return projected

    @staticmethod
    def yaw_to_transform(yaw: float) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        c, s = math.cos(yaw), math.sin(yaw)
        transform[0, 0] = c
        transform[0, 1] = -s
        transform[1, 0] = s
        transform[1, 1] = c
        return transform

    @staticmethod
    def get_yaw_from_matrix(transform: np.ndarray) -> float:
        return math.atan2(
            float(transform[1, 0]), float(transform[0, 0])
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def quaternion_from_matrix(transform: np.ndarray):
        rotation = transform[:3, :3]
        trace = float(np.trace(rotation))

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-12:
            return 0.0, 0.0, 0.0, 1.0

        return (
            float(qx / norm),
            float(qy / norm),
            float(qz / norm),
            float(qw / norm),
        )

    @staticmethod
    def stamp_to_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return float((time.perf_counter() - start_time) * 1000.0)


def main(args=None):
    rclpy.init(args=args)
    node = LivoxSlamPoseNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
