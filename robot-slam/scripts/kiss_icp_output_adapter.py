import copy
import json
import math
from collections import deque
from pathlib import Path as FilePath

import numpy as np
import rclpy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker


class KissIcpOutputAdapter(Node):
    """
    Adapter between native KISS-ICP ROS 2 topics and the common
    /g1/slam/* interface used by the TFM.

    Important frame detail:

    - /kiss/frame keeps the original PointCloud2 header/frame_id.
    - When KISS-ICP uses base_frame=livox_corrected_frame, /kiss/odometry
      is expressed for that corrected moving frame.
    - Therefore /kiss/frame must first be transformed from livox_frame to
      livox_corrected_frame before the odometry pose is applied.

    The mount transform is read from the same livox_slam_config.json used by
    the other front-ends.

    The current upstream KISS-ICP ROS wrapper also publishes /kiss/local_map
    in lidar_odom_frame without applying the cloud->base correction when a
    different base_frame is configured. This adapter can compensate for that
    behavior through kiss_icp.correct_local_map_for_base_frame_bug.
    """

    def __init__(self):
        super().__init__("kiss_icp_output_adapter")

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter("config_file", "")
        self.declare_parameter("odom_input_topic", "/kiss/odometry")
        self.declare_parameter("frame_input_topic", "/kiss/frame")
        self.declare_parameter("local_map_input_topic", "/kiss/local_map")
        self.declare_parameter("fixed_frame", "map")
        self.declare_parameter("base_frame", "livox_frame")
        self.declare_parameter("raw_lidar_frame", "livox_frame")
        self.declare_parameter("max_path_length", 5000)
        self.declare_parameter("publish_status", True)
        self.declare_parameter("publish_marker", True)
        self.declare_parameter("clear_path_on_clock_jump", True)
        self.declare_parameter("clock_jump_threshold_sec", 1.0)
        self.declare_parameter("aligned_cloud_sync_tolerance_sec", 0.02)
        self.declare_parameter("sync_queue_size", 30)

        # ============================================================
        # READ PARAMETERS
        # ============================================================

        self.config_file = str(
            self.get_parameter("config_file").value
        ).strip()

        self.odom_input_topic = str(
            self.get_parameter("odom_input_topic").value
        )
        self.frame_input_topic = str(
            self.get_parameter("frame_input_topic").value
        )
        self.local_map_input_topic = str(
            self.get_parameter("local_map_input_topic").value
        )
        self.fixed_frame = str(
            self.get_parameter("fixed_frame").value
        )
        self.base_frame = str(
            self.get_parameter("base_frame").value
        )
        self.raw_lidar_frame = str(
            self.get_parameter("raw_lidar_frame").value
        )
        self.max_path_length = int(
            self.get_parameter("max_path_length").value
        )
        self.publish_status_enabled = bool(
            self.get_parameter("publish_status").value
        )
        self.publish_marker_enabled = bool(
            self.get_parameter("publish_marker").value
        )
        self.clear_path_on_clock_jump = bool(
            self.get_parameter("clear_path_on_clock_jump").value
        )
        self.clock_jump_threshold_sec = float(
            self.get_parameter("clock_jump_threshold_sec").value
        )
        self.aligned_cloud_sync_tolerance_sec = float(
            self.get_parameter("aligned_cloud_sync_tolerance_sec").value
        )
        self.sync_queue_size = int(
            self.get_parameter("sync_queue_size").value
        )

        # ============================================================
        # CENTRAL PROJECT CONFIG / MOUNT CORRECTION
        # ============================================================

        self.config = self.load_json_config(self.config_file)

        frames = self.config.get("frames", {})
        livox_cfg = (
            self.config
            .get("sensors", {})
            .get("livox", {})
        )
        kiss_cfg = (
            self.config
            .get("slam", {})
            .get("algorithm_options", {})
            .get("kiss_icp", {})
        )

        self.corrected_lidar_frame = str(
            frames.get(
                "corrected_lidar_frame",
                "livox_corrected_frame",
            )
        )

        self.apply_mount_correction = bool(
            livox_cfg.get("apply_mount_correction", False)
        )

        translation_cfg = livox_cfg.get("translation_m", {})
        rotation_cfg = livox_cfg.get("rotation_deg", {})

        translation = np.array(
            [
                float(translation_cfg.get("x", 0.0)),
                float(translation_cfg.get("y", 0.0)),
                float(translation_cfg.get("z", 0.0)),
            ],
            dtype=np.float64,
        )

        rpy = np.deg2rad(
            np.array(
                [
                    float(rotation_cfg.get("roll", 0.0)),
                    float(rotation_cfg.get("pitch", 0.0)),
                    float(rotation_cfg.get("yaw", 0.0)),
                ],
                dtype=np.float64,
            )
        )

        self.mount_transform = self.build_transform_from_xyz_rpy(
            translation,
            rpy,
        )

        self.correct_local_map_for_base_frame_bug = bool(
            kiss_cfg.get(
                "correct_local_map_for_base_frame_bug",
                True,
            )
        )

        # T_base_sensor used before applying T_fixed_base.
        if (
            self.apply_mount_correction
            and self.base_frame == self.corrected_lidar_frame
            and self.raw_lidar_frame != self.base_frame
        ):
            self.sensor_to_base_transform = self.mount_transform.copy()
        else:
            self.sensor_to_base_transform = np.eye(
                4,
                dtype=np.float64,
            )

        # ============================================================
        # PUBLISHERS
        # ============================================================

        self.pose_pub = self.create_publisher(
            PoseStamped,
            "/g1/slam/pose",
            10,
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            "/g1/slam/odom",
            10,
        )
        self.path_pub = self.create_publisher(
            Path,
            "/g1/slam/path",
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            "/g1/slam/status",
            10,
        )
        self.marker_pub = self.create_publisher(
            Marker,
            "/g1/slam/robot_marker",
            10,
        )
        self.aligned_cloud_pub = self.create_publisher(
            PointCloud2,
            "/g1/slam/aligned_cloud",
            10,
        )
        self.local_map_pub = self.create_publisher(
            PointCloud2,
            "/g1/slam/local_map",
            10,
        )

        # ============================================================
        # SUBSCRIBERS
        # ============================================================

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_input_topic,
            self.odom_callback,
            10,
        )
        self.frame_sub = self.create_subscription(
            PointCloud2,
            self.frame_input_topic,
            self.frame_callback,
            10,
        )
        self.local_map_sub = self.create_subscription(
            PointCloud2,
            self.local_map_input_topic,
            self.local_map_callback,
            10,
        )

        # ============================================================
        # STATE
        # ============================================================

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.fixed_frame

        self.last_odom_stamp_sec = None

        self.odom_count = 0
        self.frame_count = 0
        self.aligned_cloud_count = 0
        self.local_map_count = 0
        self.local_map_points = 0

        self.warned_odom_frame_mismatch = False
        self.warned_cloud_frame_mismatch = False
        self.warned_odom_child_mismatch = False
        self.warned_map_frame_mismatch = False

        self.odom_queue = deque(maxlen=self.sync_queue_size)
        self.frame_queue = deque(maxlen=self.sync_queue_size)

        # ============================================================
        # STARTUP INFORMATION
        # ============================================================

        self.get_logger().info("KISS-ICP output adapter started")
        self.get_logger().info(
            f"Odometry input: {self.odom_input_topic}"
        )
        self.get_logger().info(
            f"Frame input: {self.frame_input_topic}"
        )
        self.get_logger().info(
            f"Local map input: {self.local_map_input_topic}"
        )
        self.get_logger().info(
            f"Project fixed frame: {self.fixed_frame}"
        )
        self.get_logger().info(
            f"KISS moving/base frame: {self.base_frame}"
        )
        self.get_logger().info(
            f"Raw LiDAR frame: {self.raw_lidar_frame}"
        )
        self.get_logger().info(
            "Mount correction in KISS adapter: "
            f"{self.apply_mount_correction}"
        )
        self.get_logger().info(
            "KISS local-map base-frame compensation: "
            f"{self.correct_local_map_for_base_frame_bug}"
        )

    # ============================================================
    # CONFIG
    # ============================================================

    @staticmethod
    def load_json_config(config_file):
        if not config_file:
            return {}

        path = FilePath(config_file).expanduser().resolve()

        if not path.exists():
            raise RuntimeError(
                f"KISS adapter config file not found: {path}"
            )

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    # ============================================================
    # TIME
    # ============================================================

    @staticmethod
    def stamp_to_seconds(stamp):
        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    # ============================================================
    # TRANSFORMS
    # ============================================================

    @staticmethod
    def build_transform_from_xyz_rpy(translation, rpy):
        roll, pitch, yaw = rpy

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float64,
        )

        rot_y = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float64,
        )

        rot_z = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot_z @ rot_y @ rot_x
        transform[:3, 3] = translation

        return transform

    @staticmethod
    def quaternion_to_rotation_matrix(x, y, z, w):
        norm = math.sqrt(
            x * x + y * y + z * z + w * w
        )

        if norm < 1e-12:
            return np.eye(3, dtype=np.float64)

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

    @classmethod
    def odom_to_matrix(cls, msg):
        """Convert nav_msgs/Odometry pose to T_fixed_moving."""

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = cls.quaternion_to_rotation_matrix(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        transform[:3, 3] = [
            position.x,
            position.y,
            position.z,
        ]

        return transform

    @staticmethod
    def transform_points(points, transform):
        if points.size == 0:
            return points

        return (
            points @ transform[:3, :3].T
            + transform[:3, 3]
        )

    # ============================================================
    # POINT CLOUD CONVERSION
    # ============================================================

    @staticmethod
    def pointcloud_to_xyz(msg):
        try:
            points = point_cloud2.read_points_numpy(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            points = np.asarray(
                points,
                dtype=np.float64,
            ).reshape(-1, 3)
        except AttributeError:
            points = np.asarray(
                list(
                    point_cloud2.read_points(
                        msg,
                        field_names=("x", "y", "z"),
                        skip_nans=True,
                    )
                ),
                dtype=np.float64,
            ).reshape(-1, 3)

        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        finite = np.all(np.isfinite(points), axis=1)
        return points[finite]

    @staticmethod
    def xyz_to_cloud(points, stamp, frame_id):
        header = Header()
        header.stamp = copy.deepcopy(stamp)
        header.frame_id = frame_id

        return point_cloud2.create_cloud_xyz32(
            header,
            points.astype(np.float32),
        )

    # ============================================================
    # RESET
    # ============================================================

    def maybe_reset(self, stamp):
        current = self.stamp_to_seconds(stamp)

        if (
            self.clear_path_on_clock_jump
            and self.last_odom_stamp_sec is not None
            and current
            < (
                self.last_odom_stamp_sec
                - self.clock_jump_threshold_sec
            )
        ):
            self.get_logger().warn(
                "Detected backward time jump. "
                "Clearing adapted KISS-ICP state."
            )

            self.path_msg = Path()
            self.path_msg.header.frame_id = self.fixed_frame
            self.odom_queue.clear()
            self.frame_queue.clear()

        self.last_odom_stamp_sec = current

    # ============================================================
    # ODOMETRY
    # ============================================================

    def odom_callback(self, msg):
        self.maybe_reset(msg.header.stamp)
        self.odom_count += 1

        if (
            msg.header.frame_id
            and msg.header.frame_id != self.fixed_frame
            and not self.warned_odom_frame_mismatch
        ):
            self.get_logger().warn(
                "KISS-ICP odometry frame is "
                f"'{msg.header.frame_id}', expected "
                f"'{self.fixed_frame}'."
            )
            self.warned_odom_frame_mismatch = True

        if (
            msg.child_frame_id
            and msg.child_frame_id != self.base_frame
            and not self.warned_odom_child_mismatch
        ):
            self.get_logger().warn(
                "KISS-ICP odometry child frame is "
                f"'{msg.child_frame_id}', expected "
                f"base frame '{self.base_frame}'."
            )
            self.warned_odom_child_mismatch = True

        # Preserve native KISS odometry unchanged.
        self.odom_pub.publish(copy.deepcopy(msg))

        pose_msg = PoseStamped()
        pose_msg.header = copy.deepcopy(msg.header)
        pose_msg.pose = copy.deepcopy(msg.pose.pose)
        self.pose_pub.publish(pose_msg)

        self.path_msg.header.stamp = copy.deepcopy(msg.header.stamp)
        self.path_msg.header.frame_id = (
            msg.header.frame_id or self.fixed_frame
        )
        self.path_msg.poses.append(copy.deepcopy(pose_msg))

        if (
            self.max_path_length > 0
            and len(self.path_msg.poses) > self.max_path_length
        ):
            self.path_msg.poses = self.path_msg.poses[
                -self.max_path_length:
            ]

        self.path_pub.publish(self.path_msg)

        if self.publish_marker_enabled:
            self.marker_pub.publish(
                self.build_marker(pose_msg)
            )

        self.odom_queue.append(
            (
                self.stamp_to_seconds(msg.header.stamp),
                copy.deepcopy(msg),
            )
        )

        self.try_publish_aligned_cloud()

        if self.publish_status_enabled:
            self.publish_status(msg)

    # ============================================================
    # CURRENT KISS FRAME
    # ============================================================

    def frame_callback(self, msg):
        self.frame_count += 1

        self.frame_queue.append(
            (
                self.stamp_to_seconds(msg.header.stamp),
                copy.deepcopy(msg),
            )
        )

        self.try_publish_aligned_cloud()

    # ============================================================
    # SYNCHRONIZE FRAME + ODOMETRY
    # ============================================================

    def try_publish_aligned_cloud(self):
        if not self.odom_queue or not self.frame_queue:
            return

        best_match = None

        for odom_index, (odom_time, _odom) in enumerate(
            self.odom_queue
        ):
            for frame_index, (frame_time, _frame) in enumerate(
                self.frame_queue
            ):
                delta = abs(odom_time - frame_time)

                if (
                    best_match is None
                    or delta < best_match[0]
                ):
                    best_match = (
                        delta,
                        odom_index,
                        frame_index,
                    )

        if best_match is None:
            return

        time_delta, odom_index, frame_index = best_match

        if time_delta > self.aligned_cloud_sync_tolerance_sec:
            return

        _odom_time, odom_msg = self.odom_queue[odom_index]
        _frame_time, frame_msg = self.frame_queue[frame_index]

        del self.odom_queue[odom_index]
        del self.frame_queue[frame_index]

        self.publish_aligned_cloud(
            odom_msg,
            frame_msg,
        )

    # ============================================================
    # ALIGNED CLOUD
    # ============================================================

    def publish_aligned_cloud(self, odom_msg, frame_msg):
        if odom_msg.header.frame_id != self.fixed_frame:
            return

        moving_frame = odom_msg.child_frame_id
        cloud_frame = frame_msg.header.frame_id

        points = self.pointcloud_to_xyz(frame_msg)

        if len(points) == 0:
            return

        # /kiss/frame retains the original LiDAR frame. When KISS odometry
        # is published for a different base_frame, convert raw cloud points
        # to the same moving frame before applying T_fixed_moving.
        if cloud_frame == moving_frame:
            points_in_moving = points
        elif (
            cloud_frame == self.raw_lidar_frame
            and moving_frame == self.base_frame
            and self.apply_mount_correction
        ):
            points_in_moving = self.transform_points(
                points,
                self.sensor_to_base_transform,
            )
        else:
            if not self.warned_cloud_frame_mismatch:
                self.get_logger().warn(
                    "/kiss/frame is expressed in "
                    f"'{cloud_frame}' while KISS odometry "
                    f"uses moving frame '{moving_frame}'. "
                    "No configured transform matches this pair; "
                    "skipping /g1/slam/aligned_cloud."
                )
                self.warned_cloud_frame_mismatch = True
            return

        fixed_from_moving = self.odom_to_matrix(odom_msg)

        aligned_points = self.transform_points(
            points_in_moving,
            fixed_from_moving,
        )

        self.aligned_cloud_pub.publish(
            self.xyz_to_cloud(
                aligned_points,
                frame_msg.header.stamp,
                self.fixed_frame,
            )
        )

        self.aligned_cloud_count += 1

    # ============================================================
    # LOCAL MAP
    # ============================================================

    def local_map_callback(self, msg):
        self.local_map_count += 1

        points = self.pointcloud_to_xyz(msg)
        self.local_map_points = int(points.shape[0])

        if len(points) == 0:
            return

        if (
            msg.header.frame_id
            and msg.header.frame_id != self.fixed_frame
            and not self.warned_map_frame_mismatch
        ):
            self.get_logger().warn(
                "KISS-ICP local map frame is "
                f"'{msg.header.frame_id}', expected "
                f"'{self.fixed_frame}'."
            )
            self.warned_map_frame_mismatch = True

        # KISS-ICP <= current upstream behavior stores the local map in the
        # LiDAR-centric odometry basis even when base_frame differs, while
        # labelling the cloud as lidar_odom_frame. Apply the same cloud->base
        # change of basis used for odometry so the normalized project map is
        # consistent with /g1/slam/odom and /g1/slam/aligned_cloud.
        if (
            self.correct_local_map_for_base_frame_bug
            and self.apply_mount_correction
            and self.base_frame != self.raw_lidar_frame
        ):
            points = self.transform_points(
                points,
                self.sensor_to_base_transform,
            )

        self.local_map_pub.publish(
            self.xyz_to_cloud(
                points,
                msg.header.stamp,
                self.fixed_frame,
            )
        )

    # ============================================================
    # MARKER
    # ============================================================

    @staticmethod
    def build_marker(pose_msg):
        marker = Marker()
        marker.header = copy.deepcopy(pose_msg.header)
        marker.ns = "g1_slam"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = copy.deepcopy(pose_msg.pose)
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.0
        marker.color.a = 1.0
        return marker

    # ============================================================
    # STATUS
    # ============================================================

    def publish_status(self, odom_msg):
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation

        status = {
            "algorithm": "kiss_icp",
            "accepted": True,
            "reason": "kiss_icp_odometry",
            "odom_messages": self.odom_count,
            "frame_messages": self.frame_count,
            "aligned_cloud_messages": self.aligned_cloud_count,
            "local_map_messages": self.local_map_count,
            "local_map_points": self.local_map_points,
            "frame_id": odom_msg.header.frame_id,
            "child_frame_id": odom_msg.child_frame_id,
            "base_frame": self.base_frame,
            "raw_lidar_frame": self.raw_lidar_frame,
            "mount_correction": self.apply_mount_correction,
            "local_map_base_frame_compensation": (
                self.correct_local_map_for_base_frame_bug
            ),
            "position": {
                "x": float(position.x),
                "y": float(position.y),
                "z": float(position.z),
            },
            "orientation": {
                "x": float(orientation.x),
                "y": float(orientation.y),
                "z": float(orientation.z),
                "w": float(orientation.w),
            },
        }

        out = String()
        out.data = json.dumps(
            status,
            separators=(",", ":"),
        )
        self.status_pub.publish(out)


def main():
    rclpy.init()

    node = KissIcpOutputAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
