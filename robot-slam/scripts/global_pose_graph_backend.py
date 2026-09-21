
"""
Common loop-closure + pose-graph backend for the TFM G1 SLAM project.

Normalized frontend inputs:
    /g1/slam/odom
    /g1/slam/aligned_cloud

Expected convention:
    /g1/slam/odom:
        header.frame_id = map
        pose = T_map_sensor

    /g1/slam/aligned_cloud:
        header.frame_id = map
        points already transformed to the global map frame

The backend is independent of the selected frontend:

    ICP propio
    KISS-ICP
    FAST-LIO2

Pipeline:
    frontend odometry
        ->
    keyframe selection
        ->
    sequential odometry edges
        ->
    loop candidate search
        ->
    ICP loop validation
        ->
    loop-closure edge
        ->
    Open3D pose-graph optimization
        ->
    optimized trajectory + optimized map


Published outputs:
    /g1/slam/optimized/pose
    /g1/slam/optimized/odom
    /g1/slam/optimized/path
    /g1/slam/optimized/global_map
    /g1/slam/optimized/status
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header
from std_msgs.msg import String


# ============================================================
# GENERAL HELPERS
# ============================================================

def stamp_to_seconds(stamp):
    return (
        float(stamp.sec)
        + float(stamp.nanosec) * 1e-9
    )


def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def yaw_from_matrix(transform):
    return math.atan2(
        float(transform[1, 0]),
        float(transform[0, 0]),
    )


# ============================================================
# ROTATION / POSE CONVERSION
# ============================================================

def quaternion_to_rotation_matrix(
    x,
    y,
    z,
    w,
):
    norm = math.sqrt(
        x * x
        + y * y
        + z * z
        + w * w
    )

    if norm < 1e-12:
        return np.eye(
            3,
            dtype=np.float64,
        )

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(
    rotation,
):
    """
    Returns quaternion as:
        x, y, z, w
    """

    trace = float(
        np.trace(rotation)
    )

    if trace > 0.0:

        s = (
            math.sqrt(
                trace + 1.0
            )
            * 2.0
        )

        w = 0.25 * s

        x = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / s

        y = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / s

        z = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / s

    elif (
        rotation[0, 0] > rotation[1, 1]
        and rotation[0, 0] > rotation[2, 2]
    ):

        s = (
            math.sqrt(
                max(
                    1.0
                    + rotation[0, 0]
                    - rotation[1, 1]
                    - rotation[2, 2],
                    0.0,
                )
            )
            * 2.0
        )

        if s < 1e-12:
            return (
                0.0,
                0.0,
                0.0,
                1.0,
            )

        w = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / s

        x = 0.25 * s

        y = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / s

        z = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / s

    elif (
        rotation[1, 1]
        > rotation[2, 2]
    ):

        s = (
            math.sqrt(
                max(
                    1.0
                    + rotation[1, 1]
                    - rotation[0, 0]
                    - rotation[2, 2],
                    0.0,
                )
            )
            * 2.0
        )

        if s < 1e-12:
            return (
                0.0,
                0.0,
                0.0,
                1.0,
            )

        w = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / s

        x = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / s

        y = 0.25 * s

        z = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / s

    else:

        s = (
            math.sqrt(
                max(
                    1.0
                    + rotation[2, 2]
                    - rotation[0, 0]
                    - rotation[1, 1],
                    0.0,
                )
            )
            * 2.0
        )

        if s < 1e-12:
            return (
                0.0,
                0.0,
                0.0,
                1.0,
            )

        w = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / s

        x = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / s

        y = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / s

        z = 0.25 * s

    quaternion = np.array(
        [
            x,
            y,
            z,
            w,
        ],
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        quaternion
    )

    if norm < 1e-12:
        return (
            0.0,
            0.0,
            0.0,
            1.0,
        )

    quaternion /= norm

    return tuple(
        float(value)
        for value in quaternion
    )


def odom_to_matrix(
    msg,
):
    """
    Convert:
        nav_msgs/Odometry

    into:
        T_map_sensor
    """

    position = (
        msg.pose.pose.position
    )

    orientation = (
        msg.pose.pose.orientation
    )

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[:3, :3] = (
        quaternion_to_rotation_matrix(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
    )

    transform[:3, 3] = [
        position.x,
        position.y,
        position.z,
    ]

    return transform


def transform_points(
    points,
    transform,
):
    if points.size == 0:
        return points.copy()

    return (
        points
        @ transform[:3, :3].T
        + transform[:3, 3]
    )


# ============================================================
# BACKEND NODE
# ============================================================

class GlobalPoseGraphBackend(Node):

    def __init__(self):

        super().__init__(
            "global_pose_graph_backend"
        )

        # ====================================================
        # CONFIGURATION
        # ====================================================

        self.declare_parameter(
            "config_file",
            "",
        )

        config_path = str(
            self.get_parameter(
                "config_file"
            ).value
        )

        self.config = (
            self.load_config(
                config_path
            )
        )

        cfg = self.config.get(
            "global_optimization",
            {},
        )

        loop_cfg = cfg.get(
            "loop_closure",
            {},
        )

        optimizer_cfg = cfg.get(
            "optimizer",
            {},
        )

        publish_cfg = cfg.get(
            "publish",
            {},
        )

        frames_cfg = self.config.get(
            "frames",
            {},
        )

        debug_cfg = self.config.get(
            "debug",
            {},
        )

        # ====================================================
        # FRAMES / TOPICS
        # ====================================================

        self.fixed_frame = str(
            frames_cfg.get(
                "fixed_frame",
                "map",
            )
        )

        self.odom_topic = str(
            cfg.get(
                "odom_topic",
                "/g1/slam/odom",
            )
        )

        self.cloud_topic = str(
            cfg.get(
                "aligned_cloud_topic",
                "/g1/slam/aligned_cloud",
            )
        )

        # ====================================================
        # SYNCHRONIZATION
        # ====================================================

        self.sync_tolerance = float(
            cfg.get(
                "sync_tolerance_sec",
                0.08,
            )
        )

        self.queue_size = int(
            cfg.get(
                "sync_queue_size",
                30,
            )
        )

        self.clock_jump_threshold = float(
            cfg.get(
                "clock_jump_threshold_sec",
                1.0,
            )
        )

        # ====================================================
        # KEYFRAMES
        # ====================================================

        self.keyframe_min_translation = float(
            loop_cfg.get(
                "keyframe_min_translation",
                0.5,
            )
        )

        self.keyframe_min_yaw = math.radians(
            float(
                loop_cfg.get(
                    "keyframe_min_yaw_deg",
                    10.0,
                )
            )
        )

        # ====================================================
        # LOOP CANDIDATE SEARCH
        # ====================================================

        self.min_keyframe_separation = int(
            loop_cfg.get(
                "min_keyframe_separation",
                25,
            )
        )

        self.search_radius = float(
            loop_cfg.get(
                "search_radius",
                2.0,
            )
        )

        self.max_candidates = int(
            loop_cfg.get(
                "max_candidates",
                5,
            )
        )

        # ====================================================
        # LOOP ICP
        # ====================================================

        self.voxel_size = float(
            loop_cfg.get(
                "voxel_size",
                0.20,
            )
        )

        self.max_correspondence_distance = float(
            loop_cfg.get(
                "max_correspondence_distance",
                0.70,
            )
        )

        self.max_iterations = int(
            loop_cfg.get(
                "max_iterations",
                80,
            )
        )

        self.min_fitness = float(
            loop_cfg.get(
                "min_fitness",
                0.35,
            )
        )

        self.max_rmse = float(
            loop_cfg.get(
                "max_rmse",
                0.30,
            )
        )

        self.max_correction_translation = float(
            loop_cfg.get(
                "max_correction_translation",
                1.5,
            )
        )

        self.max_correction_yaw = math.radians(
            float(
                loop_cfg.get(
                    "max_correction_yaw_deg",
                    45.0,
                )
            )
        )

        # ====================================================
        # POSE GRAPH OPTIMIZER
        # ====================================================

        self.edge_prune_threshold = float(
            optimizer_cfg.get(
                "edge_prune_threshold",
                0.25,
            )
        )

        self.preference_loop_closure = float(
            optimizer_cfg.get(
                "preference_loop_closure",
                2.0,
            )
        )

        self.reference_node = int(
            optimizer_cfg.get(
                "reference_node",
                0,
            )
        )

        # ====================================================
        # OUTPUT MAP
        # ====================================================

        self.publish_map_every_n_keyframes = int(
            publish_cfg.get(
                "map_every_n_keyframes",
                5,
            )
        )

        self.map_voxel_size = float(
            publish_cfg.get(
                "map_voxel_size",
                self.voxel_size,
            )
        )

        self.max_map_points = int(
            publish_cfg.get(
                "max_map_points",
                250000,
            )
        )

        self.max_path_length = int(
            debug_cfg.get(
                "max_path_length",
                5000,
            )
        )

        # ====================================================
        # STATE
        # ====================================================

        self.pose_graph = (
            o3d.pipelines.registration.PoseGraph()
        )

        self.keyframes = []

        self.loop_pairs = set()

        self.odom_queue = []

        self.cloud_queue = []

        self.last_odom_stamp = None

        self.optimizer_runs = 0

        self.loop_candidates_tested = 0

        self.loop_candidates_accepted = 0

        self.last_loop_data = None

        # ====================================================
        # QoS
        # ====================================================

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ====================================================
        # PUBLISHERS
        # ====================================================

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/g1/slam/optimized/pose",
            10,
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            "/g1/slam/optimized/odom",
            10,
        )

        self.path_pub = self.create_publisher(
            PathMsg,
            "/g1/slam/optimized/path",
            10,
        )

        self.map_pub = self.create_publisher(
            PointCloud2,
            "/g1/slam/optimized/global_map",
            map_qos,
        )

        self.status_pub = self.create_publisher(
            String,
            "/g1/slam/optimized/status",
            10,
        )

        # ====================================================
        # SUBSCRIBERS
        # ====================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            50,
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.cloud_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        # ====================================================
        # STARTUP INFO
        # ====================================================

        self.get_logger().info(
            "Global pose-graph backend started"
        )

        self.get_logger().info(
            f"Odometry input: {self.odom_topic}"
        )

        self.get_logger().info(
            f"Aligned cloud input: {self.cloud_topic}"
        )

        self.get_logger().info(
            f"Fixed frame: {self.fixed_frame}"
        )

        self.get_logger().info(
            "Loop closure strategy: "
            "spatial candidate search + point-to-point ICP"
        )

        self.get_logger().info(
            "Publishing optimized outputs under "
            "/g1/slam/optimized/*"
        )

    # ============================================================
    # CONFIG
    # ============================================================

    def load_config(
        self,
        config_path,
    ):

        if not config_path:
            return {}

        path = Path(
            config_path
        ).expanduser()

        if not path.is_absolute():

            path = (
                Path.cwd()
                / path
            )

        if not path.exists():

            raise RuntimeError(
                f"Config file not found: {path}"
            )

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        self.get_logger().warn(
            "Resetting global pose graph"
        )

        self.pose_graph = (
            o3d.pipelines.registration.PoseGraph()
        )

        self.keyframes.clear()

        self.loop_pairs.clear()

        self.odom_queue.clear()

        self.cloud_queue.clear()

        self.optimizer_runs = 0

        self.loop_candidates_tested = 0

        self.loop_candidates_accepted = 0

        self.last_loop_data = None

    # ============================================================
    # INPUT CALLBACKS
    # ============================================================

    def odom_callback(
        self,
        msg,
    ):

        stamp = stamp_to_seconds(
            msg.header.stamp
        )

        if (
            self.last_odom_stamp is not None
            and stamp
            < (
                self.last_odom_stamp
                - self.clock_jump_threshold
            )
        ):

            self.reset()

        self.last_odom_stamp = stamp

        self.odom_queue.append(
            (
                stamp,
                msg,
            )
        )

        self.odom_queue = (
            self.odom_queue[
                -self.queue_size:
            ]
        )

        self.try_match()

    def cloud_callback(
        self,
        msg,
    ):

        stamp = stamp_to_seconds(
            msg.header.stamp
        )

        self.cloud_queue.append(
            (
                stamp,
                msg,
            )
        )

        self.cloud_queue = (
            self.cloud_queue[
                -self.queue_size:
            ]
        )

        self.try_match()

    # ============================================================
    # ODOM / CLOUD SYNCHRONIZATION
    # ============================================================

    def try_match(self):

        if (
            not self.odom_queue
            or not self.cloud_queue
        ):
            return

        best_match = None

        for odom_index, (
            odom_time,
            _,
        ) in enumerate(
            self.odom_queue
        ):

            for cloud_index, (
                cloud_time,
                _,
            ) in enumerate(
                self.cloud_queue
            ):

                delta = abs(
                    odom_time
                    - cloud_time
                )

                if (
                    best_match is None
                    or delta < best_match[0]
                ):

                    best_match = (
                        delta,
                        odom_index,
                        cloud_index,
                    )

        if best_match is None:
            return

        (
            delta,
            odom_index,
            cloud_index,
        ) = best_match

        if (
            delta
            > self.sync_tolerance
        ):
            return

        _, odom_msg = (
            self.odom_queue.pop(
                odom_index
            )
        )

        _, cloud_msg = (
            self.cloud_queue.pop(
                cloud_index
            )
        )

        try:

            self.process_observation(
                odom_msg,
                cloud_msg,
            )

        except Exception as exc:

            self.get_logger().error(
                "Backend observation failed: "
                f"{exc}"
            )

    # ============================================================
    # POINT CLOUD
    # ============================================================

    def pointcloud_to_xyz(
        self,
        msg,
    ):

        try:

            points = (
                point_cloud2.read_points_numpy(
                    msg,
                    field_names=(
                        "x",
                        "y",
                        "z",
                    ),
                    skip_nans=True,
                )
            )

            points = np.asarray(
                points,
                dtype=np.float64,
            ).reshape(
                -1,
                3,
            )

        except AttributeError:

            points = np.asarray(
                list(
                    point_cloud2.read_points(
                        msg,
                        field_names=(
                            "x",
                            "y",
                            "z",
                        ),
                        skip_nans=True,
                    )
                ),
                dtype=np.float64,
            ).reshape(
                -1,
                3,
            )

        if points.size == 0:

            return np.empty(
                (
                    0,
                    3,
                ),
                dtype=np.float64,
            )

        finite = np.all(
            np.isfinite(points),
            axis=1,
        )

        return points[
            finite
        ]

    def create_cloud(
        self,
        points,
    ):

        cloud = (
            o3d.geometry.PointCloud()
        )

        cloud.points = (
            o3d.utility.Vector3dVector(
                points
            )
        )

        if self.voxel_size > 0.0:

            cloud = (
                cloud.voxel_down_sample(
                    self.voxel_size
                )
            )

        return cloud

    # ============================================================
    # KEYFRAME SELECTION
    # ============================================================

    def should_add_keyframe(
        self,
        raw_pose,
    ):

        if not self.keyframes:
            return True

        previous_pose = (
            self.keyframes[-1][
                "raw_pose"
            ]
        )

        translation = float(
            np.linalg.norm(
                raw_pose[:3, 3]
                - previous_pose[:3, 3]
            )
        )

        yaw_delta = abs(
            normalize_angle(
                yaw_from_matrix(
                    raw_pose
                )
                - yaw_from_matrix(
                    previous_pose
                )
            )
        )

        return (
            translation
            >= self.keyframe_min_translation
            or yaw_delta
            >= self.keyframe_min_yaw
        )

    # ============================================================
    # PROCESS OBSERVATION
    # ============================================================

    def process_observation(
        self,
        odom_msg,
        cloud_msg,
    ):

        # --------------------------------------------------------
        # Validate frames
        # --------------------------------------------------------

        if (
            odom_msg.header.frame_id
            and odom_msg.header.frame_id
            != self.fixed_frame
        ):

            self.get_logger().warn(
                "Ignoring odometry in frame "
                f"'{odom_msg.header.frame_id}'. "
                f"Expected '{self.fixed_frame}'."
            )

            return

        if (
            cloud_msg.header.frame_id
            and cloud_msg.header.frame_id
            != self.fixed_frame
        ):

            self.get_logger().warn(
                "Ignoring aligned cloud in frame "
                f"'{cloud_msg.header.frame_id}'. "
                f"Expected '{self.fixed_frame}'."
            )

            return

        raw_pose = odom_to_matrix(
            odom_msg
        )

        if not self.should_add_keyframe(
            raw_pose
        ):
            return

        points_global = (
            self.pointcloud_to_xyz(
                cloud_msg
            )
        )

        if len(points_global) < 100:
            return

        # --------------------------------------------------------
        # Convert global aligned cloud back to keyframe-local
        # coordinates.
        #
        # Frontend provides:
        #
        #   p_map = T_map_sensor * p_sensor
        #
        # Therefore:
        #
        #   p_sensor = inv(T_map_sensor) * p_map
        #
        # This gives the backend a frontend-independent local cloud.
        # --------------------------------------------------------

        points_local = (
            transform_points(
                points_global,
                np.linalg.inv(
                    raw_pose
                ),
            )
        )

        cloud_local = (
            self.create_cloud(
                points_local
            )
        )

        if (
            len(cloud_local.points)
            < 100
        ):
            return

        self.add_keyframe(
            raw_pose,
            cloud_local,
            odom_msg.header.stamp,
            odom_msg.child_frame_id,
        )

    # ============================================================
    # ADD KEYFRAME
    # ============================================================

    def add_keyframe(
        self,
        raw_pose,
        cloud_local,
        stamp,
        child_frame_id,
    ):

        index = len(
            self.keyframes
        )

        # --------------------------------------------------------
        # First node
        # --------------------------------------------------------

        if index == 0:

            initial_pose = (
                raw_pose.copy()
            )

            self.pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    initial_pose
                )
            )

        # --------------------------------------------------------
        # Sequential odometry edge
        # --------------------------------------------------------

        else:

            previous_raw_pose = (
                self.keyframes[-1][
                    "raw_pose"
                ]
            )

            previous_optimized_pose = (
                np.asarray(
                    self.pose_graph.nodes[
                        -1
                    ].pose
                ).copy()
            )

            # ----------------------------------------------------
            # Open3D edge convention:
            #
            # edge transformation maps:
            #
            #     source frame -> target frame
            #
            # We add:
            #
            #     previous -> current
            #
            # raw poses are:
            #
            #     T_map_previous
            #     T_map_current
            #
            # Therefore:
            #
            #     T_current_previous
            #       = inv(T_map_current)
            #         * T_map_previous
            # ----------------------------------------------------

            odom_edge = (
                np.linalg.inv(
                    raw_pose
                )
                @ previous_raw_pose
            )

            # Continue from the optimized previous node while preserving
            # the relative motion measured by the frontend.

            initial_pose = (
                previous_optimized_pose
                @ np.linalg.inv(
                    odom_edge
                )
            )

            self.pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    initial_pose
                )
            )

            previous_cloud = (
                self.keyframes[-1][
                    "cloud"
                ]
            )

            information = (
                self.information_matrix(
                    previous_cloud,
                    cloud_local,
                    odom_edge,
                )
            )

            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    index - 1,
                    index,
                    odom_edge,
                    information,
                    uncertain=False,
                )
            )

        # --------------------------------------------------------
        # Store keyframe
        # --------------------------------------------------------

        self.keyframes.append(
            {
                "raw_pose": (
                    raw_pose.copy()
                ),

                "cloud": (
                    cloud_local
                ),

                "stamp": (
                    copy.deepcopy(
                        stamp
                    )
                ),

                "child_frame_id": (
                    child_frame_id
                ),
            }
        )

        # --------------------------------------------------------
        # Search for loop closure
        # --------------------------------------------------------

        loop_result = (
            self.detect_loop(
                index
            )
        )

        loop_accepted = False

        if loop_result is not None:

            (
                candidate_index,
                registration,
                information,
                correction_translation,
                correction_yaw,
            ) = loop_result

            pair = (
                min(
                    index,
                    candidate_index,
                ),
                max(
                    index,
                    candidate_index,
                ),
            )

            if (
                pair
                not in self.loop_pairs
            ):

                self.pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        index,
                        candidate_index,
                        registration.transformation,
                        information,
                        uncertain=True,
                    )
                )

                self.loop_pairs.add(
                    pair
                )

                self.loop_candidates_accepted += 1

                self.last_loop_data = {
                    "current_keyframe": int(
                        index
                    ),

                    "matched_keyframe": int(
                        candidate_index
                    ),

                    "fitness": float(
                        registration.fitness
                    ),

                    "rmse": float(
                        registration.inlier_rmse
                    ),

                    "correction_translation": float(
                        correction_translation
                    ),

                    "correction_yaw_deg": float(
                        math.degrees(
                            correction_yaw
                        )
                    ),
                }

                self.get_logger().info(
                    "LOOP CLOSURE ACCEPTED: "
                    f"{index} <-> {candidate_index} | "
                    f"fitness={registration.fitness:.3f} | "
                    f"rmse={registration.inlier_rmse:.3f} | "
                    "translation correction="
                    f"{correction_translation:.3f} m | "
                    "yaw correction="
                    f"{math.degrees(correction_yaw):.2f} deg"
                )

                self.optimize_pose_graph()

                loop_accepted = True

        # --------------------------------------------------------
        # Publish optimized outputs
        # --------------------------------------------------------

        self.publish_optimized_pose_and_odom(
            stamp
        )

        self.publish_optimized_path(
            stamp
        )

        if (
            loop_accepted
            or self.publish_map_every_n_keyframes <= 1
            or (
                index
                % self.publish_map_every_n_keyframes
                == 0
            )
        ):

            self.publish_optimized_map(
                stamp
            )

        self.publish_status(
            loop_accepted
        )

    # ============================================================
    # INFORMATION MATRIX
    # ============================================================

    def information_matrix(
        self,
        source,
        target,
        transformation,
    ):

        try:

            return (
                o3d.pipelines.registration
                .get_information_matrix_from_point_clouds(
                    source,
                    target,
                    self.max_correspondence_distance,
                    transformation,
                )
            )

        except Exception:

            return np.eye(
                6,
                dtype=np.float64,
            )

    # ============================================================
    # LOOP CANDIDATES
    # ============================================================

    def candidate_indices(
        self,
        current_index,
    ):

        if (
            current_index
            < self.min_keyframe_separation
        ):
            return []

        current_pose = np.asarray(
            self.pose_graph.nodes[
                current_index
            ].pose
        )

        current_position = (
            current_pose[
                :3,
                3
            ]
        )

        candidates = []

        maximum_index = (
            current_index
            - self.min_keyframe_separation
        )

        for candidate_index in range(
            maximum_index + 1
        ):

            candidate_pose = np.asarray(
                self.pose_graph.nodes[
                    candidate_index
                ].pose
            )

            candidate_position = (
                candidate_pose[
                    :3,
                    3
                ]
            )

            distance = float(
                np.linalg.norm(
                    candidate_position
                    - current_position
                )
            )

            if (
                distance
                > self.search_radius
            ):
                continue

            pair = (
                min(
                    candidate_index,
                    current_index,
                ),
                max(
                    candidate_index,
                    current_index,
                ),
            )

            if (
                pair
                in self.loop_pairs
            ):
                continue

            candidates.append(
                (
                    distance,
                    candidate_index,
                )
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        return [
            candidate_index
            for _, candidate_index
            in candidates[
                :self.max_candidates
            ]
        ]

    # ============================================================
    # LOOP DETECTION
    # ============================================================

    def detect_loop(
        self,
        current_index,
    ):

        candidates = (
            self.candidate_indices(
                current_index
            )
        )

        if not candidates:
            return None

        current_pose = (
            np.asarray(
                self.pose_graph.nodes[
                    current_index
                ].pose
            ).copy()
        )

        source_cloud = (
            self.keyframes[
                current_index
            ][
                "cloud"
            ]
        )

        best_result = None

        for candidate_index in candidates:

            self.loop_candidates_tested += 1

            candidate_pose = (
                np.asarray(
                    self.pose_graph.nodes[
                        candidate_index
                    ].pose
                ).copy()
            )

            target_cloud = (
                self.keyframes[
                    candidate_index
                ][
                    "cloud"
                ]
            )

            # ----------------------------------------------------
            # Initial guess:
            #
            # current local frame -> candidate local frame
            #
            # node poses are:
            #
            #   T_map_current
            #   T_map_candidate
            #
            # therefore:
            #
            #   T_candidate_current
            #       = inv(T_map_candidate)
            #         * T_map_current
            # ----------------------------------------------------

            initial_guess = (
                np.linalg.inv(
                    candidate_pose
                )
                @ current_pose
            )

            registration = (
                o3d.pipelines.registration.registration_icp(
                    source_cloud,
                    target_cloud,
                    self.max_correspondence_distance,
                    initial_guess,

                    o3d.pipelines.registration
                    .TransformationEstimationPointToPoint(),

                    o3d.pipelines.registration
                    .ICPConvergenceCriteria(
                        max_iteration=(
                            self.max_iterations
                        )
                    ),
                )
            )

            # ----------------------------------------------------
            # Registration quality gates
            # ----------------------------------------------------

            if (
                registration.fitness
                < self.min_fitness
            ):
                continue

            if (
                registration.inlier_rmse
                > self.max_rmse
            ):
                continue

            # ----------------------------------------------------
            # Plausibility gate
            #
            # Compare refined transform against the odometry-based initial
            # estimate. A loop closure that requires an implausibly large
            # correction is rejected.
            # ----------------------------------------------------

            correction = (
                registration.transformation
                @ np.linalg.inv(
                    initial_guess
                )
            )

            correction_translation = float(
                np.linalg.norm(
                    correction[
                        :3,
                        3
                    ]
                )
            )

            correction_yaw = abs(
                normalize_angle(
                    yaw_from_matrix(
                        correction
                    )
                )
            )

            if (
                correction_translation
                > self.max_correction_translation
            ):
                continue

            if (
                correction_yaw
                > self.max_correction_yaw
            ):
                continue

            information = (
                self.information_matrix(
                    source_cloud,
                    target_cloud,
                    registration.transformation,
                )
            )

            # Prefer:
            #
            #   high fitness
            #   low RMSE
            #
            # This score is used only to select the best candidate among
            # candidates that already passed all acceptance gates.

            score = (
                float(
                    registration.fitness
                )
                / max(
                    float(
                        registration.inlier_rmse
                    ),
                    1e-6,
                )
            )

            if (
                best_result is None
                or score
                > best_result[0]
            ):

                best_result = (
                    score,
                    candidate_index,
                    registration,
                    information,
                    correction_translation,
                    correction_yaw,
                )

        if best_result is None:
            return None

        (
            _score,
            candidate_index,
            registration,
            information,
            correction_translation,
            correction_yaw,
        ) = best_result

        return (
            candidate_index,
            registration,
            information,
            correction_translation,
            correction_yaw,
        )

    # ============================================================
    # GLOBAL OPTIMIZATION
    # ============================================================

    def optimize_pose_graph(
        self,
    ):

        if (
            len(
                self.pose_graph.nodes
            )
            < 2
        ):
            return

        option = (
            o3d.pipelines.registration
            .GlobalOptimizationOption(
                max_correspondence_distance=(
                    self.max_correspondence_distance
                ),

                edge_prune_threshold=(
                    self.edge_prune_threshold
                ),

                preference_loop_closure=(
                    self.preference_loop_closure
                ),

                reference_node=(
                    self.reference_node
                ),
            )
        )

        self.get_logger().info(
            "Running global pose-graph optimization..."
        )

        o3d.pipelines.registration.global_optimization(
            self.pose_graph,

            o3d.pipelines.registration
            .GlobalOptimizationLevenbergMarquardt(),

            o3d.pipelines.registration
            .GlobalOptimizationConvergenceCriteria(),

            option,
        )

        self.optimizer_runs += 1

        self.get_logger().info(
            "Global pose-graph optimization finished"
        )

    # ============================================================
    # OPTIMIZED POSE / ODOM
    # ============================================================

    def publish_optimized_pose_and_odom(
        self,
        stamp,
    ):

        if not self.pose_graph.nodes:
            return

        index = (
            len(
                self.pose_graph.nodes
            )
            - 1
        )

        transform = np.asarray(
            self.pose_graph.nodes[
                index
            ].pose
        )

        # --------------------------------------------------------
        # PoseStamped
        # --------------------------------------------------------

        pose_msg = PoseStamped()

        pose_msg.header.frame_id = (
            self.fixed_frame
        )

        pose_msg.header.stamp = (
            copy.deepcopy(
                stamp
            )
        )

        pose_msg.pose.position.x = float(
            transform[0, 3]
        )

        pose_msg.pose.position.y = float(
            transform[1, 3]
        )

        pose_msg.pose.position.z = float(
            transform[2, 3]
        )

        (
            x,
            y,
            z,
            w,
        ) = rotation_matrix_to_quaternion(
            transform[
                :3,
                :3
            ]
        )

        pose_msg.pose.orientation.x = x
        pose_msg.pose.orientation.y = y
        pose_msg.pose.orientation.z = z
        pose_msg.pose.orientation.w = w

        self.pose_pub.publish(
            pose_msg
        )

        # --------------------------------------------------------
        # Odometry
        # --------------------------------------------------------

        odom_msg = Odometry()

        odom_msg.header = (
            copy.deepcopy(
                pose_msg.header
            )
        )

        odom_msg.child_frame_id = (
            self.keyframes[
                index
            ].get(
                "child_frame_id",
                "",
            )
        )

        odom_msg.pose.pose = (
            copy.deepcopy(
                pose_msg.pose
            )
        )

        self.odom_pub.publish(
            odom_msg
        )

    # ============================================================
    # OPTIMIZED PATH
    # ============================================================

    def publish_optimized_path(
        self,
        stamp,
    ):

        msg = PathMsg()

        msg.header.frame_id = (
            self.fixed_frame
        )

        msg.header.stamp = (
            copy.deepcopy(
                stamp
            )
        )

        nodes = (
            self.pose_graph.nodes
        )

        if self.max_path_length > 0:

            start_index = max(
                0,
                len(nodes)
                - self.max_path_length,
            )

        else:

            start_index = 0

        for index in range(
            start_index,
            len(nodes),
        ):

            transform = np.asarray(
                nodes[
                    index
                ].pose
            )

            pose = PoseStamped()

            pose.header.frame_id = (
                self.fixed_frame
            )

            pose.header.stamp = (
                copy.deepcopy(
                    self.keyframes[
                        index
                    ][
                        "stamp"
                    ]
                )
            )

            pose.pose.position.x = float(
                transform[0, 3]
            )

            pose.pose.position.y = float(
                transform[1, 3]
            )

            pose.pose.position.z = float(
                transform[2, 3]
            )

            (
                x,
                y,
                z,
                w,
            ) = rotation_matrix_to_quaternion(
                transform[
                    :3,
                    :3
                ]
            )

            pose.pose.orientation.x = x
            pose.pose.orientation.y = y
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w

            msg.poses.append(
                pose
            )

        self.path_pub.publish(
            msg
        )

    # ============================================================
    # OPTIMIZED MAP
    # ============================================================

    def publish_optimized_map(
        self,
        stamp,
    ):

        all_points = []

        for index, keyframe in enumerate(
            self.keyframes
        ):

            local_points = np.asarray(
                keyframe[
                    "cloud"
                ].points
            )

            if local_points.size == 0:
                continue

            optimized_pose = np.asarray(
                self.pose_graph.nodes[
                    index
                ].pose
            )

            global_points = (
                transform_points(
                    local_points,
                    optimized_pose,
                )
            )

            all_points.append(
                global_points
            )

        if not all_points:
            return

        points = np.vstack(
            all_points
        )

        map_cloud = (
            o3d.geometry.PointCloud()
        )

        map_cloud.points = (
            o3d.utility.Vector3dVector(
                points
            )
        )

        if (
            self.map_voxel_size
            > 0.0
        ):

            map_cloud = (
                map_cloud.voxel_down_sample(
                    self.map_voxel_size
                )
            )

        points = np.asarray(
            map_cloud.points
        )

        # --------------------------------------------------------
        # Deterministic point limit
        #
        # Important for reproducible TFM experiments.
        # --------------------------------------------------------

        if (
            self.max_map_points > 0
            and len(points)
            > self.max_map_points
        ):

            indices = np.linspace(
                0,
                len(points) - 1,
                self.max_map_points,
                dtype=np.int64,
            )

            points = (
                points[
                    indices
                ]
            )

        header = Header()

        header.frame_id = (
            self.fixed_frame
        )

        header.stamp = (
            copy.deepcopy(
                stamp
            )
        )

        msg = (
            point_cloud2.create_cloud_xyz32(
                header,
                points.astype(
                    np.float32
                ),
            )
        )

        self.map_pub.publish(
            msg
        )

    # ============================================================
    # STATUS
    # ============================================================

    def publish_status(
        self,
        loop_accepted,
    ):

        status = {
            "backend": (
                "pose_graph"
            ),

            "keyframes": (
                len(
                    self.keyframes
                )
            ),

            "nodes": (
                len(
                    self.pose_graph.nodes
                )
            ),

            "edges": (
                len(
                    self.pose_graph.edges
                )
            ),

            "loop_closures": (
                len(
                    self.loop_pairs
                )
            ),

            "loop_candidates_tested": (
                self.loop_candidates_tested
            ),

            "loop_candidates_accepted": (
                self.loop_candidates_accepted
            ),

            "optimizer_runs": (
                self.optimizer_runs
            ),

            "optimized": (
                self.optimizer_runs > 0
            ),

            "loop_accepted_this_keyframe": (
                bool(
                    loop_accepted
                )
            ),
        }

        if (
            self.last_loop_data
            is not None
        ):

            status[
                "last_loop"
            ] = (
                self.last_loop_data
            )

        msg = String()

        msg.data = json.dumps(
            status,
            separators=(
                ",",
                ":",
            ),
        )

        self.status_pub.publish(
            msg
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = (
        GlobalPoseGraphBackend()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":
    main()