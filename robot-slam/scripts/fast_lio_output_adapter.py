import copy
import json
import math
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


class FastLioOutputAdapter(Node):
    """
    Normalize FAST-LIO2 outputs to the common G1 SLAM interface.

    Native FAST-LIO2 outputs:
      /Odometry
      /path
      /cloud_registered
      /Laser_map

    Common outputs:
      /g1/slam/pose
      /g1/slam/odom
      /g1/slam/path
      /g1/slam/status
      /g1/slam/aligned_cloud
      /g1/slam/local_map

    Dynamic TF:
      map -> livox_corrected_frame

    Static mounting TF:
      livox_corrected_frame -> livox_frame

    The static TF remains owned exclusively by:
      livox_mount_tf_publisher.py

    FAST-LIO2 works internally using the native Livox/IMU coordinate
    convention. This adapter converts FAST-LIO2 outputs to the corrected
    project coordinate system using the mounting correction defined in:

      config/livox_slam_config.json

    Specifically:

      sensors.livox.translation_m
      sensors.livox.rotation_deg

    Pose change of basis:

      T_corrected = C * T_native * C^-1

    Point clouds already expressed in FAST-LIO2's world frame are converted
    using:

      p_corrected = C * p_native
    """

    def __init__(self):
        super().__init__("fast_lio_output_adapter")

        # ============================================================
        # PARAMETERS
        # ============================================================

        self.declare_parameter("fixed_frame", "map")

        self.declare_parameter(
            "internal_fixed_frame",
            "camera_init",
        )

        self.declare_parameter(
            "body_frame",
            "body",
        )

        self.declare_parameter(
            "lidar_frame",
            "livox_frame",
        )

        self.declare_parameter(
            "corrected_lidar_frame",
            "livox_corrected_frame",
        )

        self.declare_parameter(
            "fast_lio_config",
            "",
        )

        # Main project configuration:
        # config/livox_slam_config.json
        self.declare_parameter(
            "config_file",
            "",
        )

        self.declare_parameter(
            "max_path_length",
            5000,
        )

        self.declare_parameter(
            "publish_tf",
            True,
        )

        self.declare_parameter(
            "publish_status",
            True,
        )

        self.fixed_frame = str(
            self.get_parameter(
                "fixed_frame"
            ).value
        )

        self.internal_fixed_frame = str(
            self.get_parameter(
                "internal_fixed_frame"
            ).value
        )

        self.body_frame = str(
            self.get_parameter(
                "body_frame"
            ).value
        )

        self.lidar_frame = str(
            self.get_parameter(
                "lidar_frame"
            ).value
        )

        self.corrected_lidar_frame = str(
            self.get_parameter(
                "corrected_lidar_frame"
            ).value
        )

        self.fast_lio_config = str(
            self.get_parameter(
                "fast_lio_config"
            ).value
        )

        self.config_file = str(
            self.get_parameter(
                "config_file"
            ).value
        ).strip()

        self.max_path_length = int(
            self.get_parameter(
                "max_path_length"
            ).value
        )

        self.publish_tf = bool(
            self.get_parameter(
                "publish_tf"
            ).value
        )

        self.publish_status_enabled = bool(
            self.get_parameter(
                "publish_status"
            ).value
        )

        # ============================================================
        # LOAD PROJECT MOUNT CONFIGURATION
        # ============================================================

        if not self.config_file:
            raise RuntimeError(
                "Parameter 'config_file' is required. "
                "It must point to livox_slam_config.json."
            )

        config_path = Path(
            self.config_file
        ).expanduser().resolve()

        if not config_path.exists():
            raise RuntimeError(
                f"Configuration file not found: {config_path}"
            )

        project_config = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        livox_cfg = (
            project_config
            .get("sensors", {})
            .get("livox", {})
        )

        self.apply_mount_correction = bool(
            livox_cfg.get(
                "apply_mount_correction",
                False,
            )
        )

        translation_cfg = livox_cfg.get(
            "translation_m",
            {},
        )

        rotation_cfg = livox_cfg.get(
            "rotation_deg",
            {},
        )

        self.mount_translation = np.array(
            [
                float(
                    translation_cfg.get(
                        "x",
                        0.0,
                    )
                ),
                float(
                    translation_cfg.get(
                        "y",
                        0.0,
                    )
                ),
                float(
                    translation_cfg.get(
                        "z",
                        0.0,
                    )
                ),
            ],
            dtype=np.float64,
        )

        self.mount_rotation_deg = (
            float(
                rotation_cfg.get(
                    "roll",
                    0.0,
                )
            ),
            float(
                rotation_cfg.get(
                    "pitch",
                    0.0,
                )
            ),
            float(
                rotation_cfg.get(
                    "yaw",
                    0.0,
                )
            ),
        )

        roll = math.radians(
            self.mount_rotation_deg[0]
        )

        pitch = math.radians(
            self.mount_rotation_deg[1]
        )

        yaw = math.radians(
            self.mount_rotation_deg[2]
        )

        self.mount_transform = np.eye(
            4,
            dtype=np.float64,
        )

        self.mount_transform[
            :3,
            :3
        ] = self.rotation_matrix_from_rpy(
            roll,
            pitch,
            yaw,
        )

        self.mount_transform[
            :3,
            3
        ] = self.mount_translation

        self.mount_transform_inv = (
            np.linalg.inv(
                self.mount_transform
            )
        )

        # ============================================================
        # STATE
        # ============================================================

        self.odom_count = 0
        self.path_count = 0
        self.aligned_cloud_count = 0
        self.local_map_count = 0

        self.last_pose = None

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
            PathMsg,
            "/g1/slam/path",
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            "/g1/slam/status",
            10,
        )

        self.aligned_pub = self.create_publisher(
            PointCloud2,
            "/g1/slam/aligned_cloud",
            10,
        )

        self.map_pub = self.create_publisher(
            PointCloud2,
            "/g1/slam/local_map",
            10,
        )

        # Dynamic TF only.
        #
        # Static Livox mount correction remains owned by:
        # livox_mount_tf_publisher.py
        self.tf_broadcaster = (
            TransformBroadcaster(
                self
            )
        )

        # ============================================================
        # SUBSCRIPTIONS
        # ============================================================

        self.create_subscription(
            Odometry,
            "/Odometry",
            self.odom_callback,
            20,
        )

        self.create_subscription(
            PathMsg,
            "/path",
            self.path_callback,
            20,
        )

        self.create_subscription(
            PointCloud2,
            "/cloud_registered",
            self.aligned_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            PointCloud2,
            "/Laser_map",
            self.map_callback,
            qos_profile_sensor_data,
        )

        # ============================================================
        # STARTUP LOG
        # ============================================================

        self.get_logger().info(
            "FAST-LIO2 common-output adapter started"
        )

        self.get_logger().info(
            "FAST-LIO2 internal fixed frame: "
            f"{self.internal_fixed_frame}"
        )

        self.get_logger().info(
            f"FAST-LIO2 body frame: "
            f"{self.body_frame}"
        )

        self.get_logger().info(
            f"Raw LiDAR frame: "
            f"{self.lidar_frame}"
        )

        self.get_logger().info(
            f"Common fixed frame: "
            f"{self.fixed_frame}"
        )

        self.get_logger().info(
            "Common moving frame: "
            f"{self.corrected_lidar_frame}"
        )

        self.get_logger().info(
            "Dynamic TF: "
            f"{self.fixed_frame} -> "
            f"{self.corrected_lidar_frame}"
        )

        self.get_logger().info(
            "Static Livox mount TF delegated to "
            "livox_mount_tf_publisher.py"
        )

        self.get_logger().info(
            "Mount correction: "
            f"enabled={self.apply_mount_correction}; "
            f"translation={self.mount_translation.tolist()}; "
            f"rotation_deg={self.mount_rotation_deg}"
        )

    # ============================================================
    # ROTATION / TRANSFORM UTILITIES
    # ============================================================

    @staticmethod
    def rotation_matrix_from_rpy(
        roll,
        pitch,
        yaw,
    ):
        """
        ROS-style fixed-axis roll/pitch/yaw rotation.

        R = Rz(yaw) * Ry(pitch) * Rx(roll)
        """

        cr = math.cos(roll)
        sr = math.sin(roll)

        cp = math.cos(pitch)
        sp = math.sin(pitch)

        cy = math.cos(yaw)
        sy = math.sin(yaw)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=np.float64,
        )

        ry = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=np.float64,
        )

        rz = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        return (
            rz
            @ ry
            @ rx
        )

    @staticmethod
    def quaternion_to_matrix(
        quaternion,
    ):
        x = float(
            quaternion.x
        )

        y = float(
            quaternion.y
        )

        z = float(
            quaternion.z
        )

        w = float(
            quaternion.w
        )

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
                    1.0 - 2.0 * (
                        y * y + z * z
                    ),
                    2.0 * (
                        x * y - z * w
                    ),
                    2.0 * (
                        x * z + y * w
                    ),
                ],
                [
                    2.0 * (
                        x * y + z * w
                    ),
                    1.0 - 2.0 * (
                        x * x + z * z
                    ),
                    2.0 * (
                        y * z - x * w
                    ),
                ],
                [
                    2.0 * (
                        x * z - y * w
                    ),
                    2.0 * (
                        y * z + x * w
                    ),
                    1.0 - 2.0 * (
                        x * x + y * y
                    ),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def matrix_to_quaternion(
        rotation,
    ):
        """
        Convert a 3x3 rotation matrix to quaternion:
        (x, y, z, w)
        """

        r = rotation

        trace = (
            r[0, 0]
            + r[1, 1]
            + r[2, 2]
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
                r[2, 1]
                - r[1, 2]
            ) / s

            y = (
                r[0, 2]
                - r[2, 0]
            ) / s

            z = (
                r[1, 0]
                - r[0, 1]
            ) / s

        elif (
            r[0, 0] > r[1, 1]
            and r[0, 0] > r[2, 2]
        ):
            s = (
                math.sqrt(
                    1.0
                    + r[0, 0]
                    - r[1, 1]
                    - r[2, 2]
                )
                * 2.0
            )

            w = (
                r[2, 1]
                - r[1, 2]
            ) / s

            x = 0.25 * s

            y = (
                r[0, 1]
                + r[1, 0]
            ) / s

            z = (
                r[0, 2]
                + r[2, 0]
            ) / s

        elif r[1, 1] > r[2, 2]:
            s = (
                math.sqrt(
                    1.0
                    + r[1, 1]
                    - r[0, 0]
                    - r[2, 2]
                )
                * 2.0
            )

            w = (
                r[0, 2]
                - r[2, 0]
            ) / s

            x = (
                r[0, 1]
                + r[1, 0]
            ) / s

            y = 0.25 * s

            z = (
                r[1, 2]
                + r[2, 1]
            ) / s

        else:
            s = (
                math.sqrt(
                    1.0
                    + r[2, 2]
                    - r[0, 0]
                    - r[1, 1]
                )
                * 2.0
            )

            w = (
                r[1, 0]
                - r[0, 1]
            ) / s

            x = (
                r[0, 2]
                + r[2, 0]
            ) / s

            y = (
                r[1, 2]
                + r[2, 1]
            ) / s

            z = 0.25 * s

        norm = math.sqrt(
            x * x
            + y * y
            + z * z
            + w * w
        )

        if norm < 1e-12:
            return (
                0.0,
                0.0,
                0.0,
                1.0,
            )

        return (
            x / norm,
            y / norm,
            z / norm,
            w / norm,
        )

    # ============================================================
    # POSE CORRECTION
    # ============================================================

    def transform_pose(
        self,
        native_pose,
    ):
        """
        Change FAST-LIO2 pose representation from the native sensor
        coordinate system to the corrected project coordinate system.

        T_corrected = C * T_native * C^-1
        """

        if not self.apply_mount_correction:
            return copy.deepcopy(
                native_pose
            )

        t_native = np.eye(
            4,
            dtype=np.float64,
        )

        t_native[
            :3,
            :3
        ] = self.quaternion_to_matrix(
            native_pose.orientation
        )

        t_native[
            :3,
            3
        ] = np.array(
            [
                native_pose.position.x,
                native_pose.position.y,
                native_pose.position.z,
            ],
            dtype=np.float64,
        )

        t_corrected = (
            self.mount_transform
            @ t_native
            @ self.mount_transform_inv
        )

        out = copy.deepcopy(
            native_pose
        )

        out.position.x = float(
            t_corrected[0, 3]
        )

        out.position.y = float(
            t_corrected[1, 3]
        )

        out.position.z = float(
            t_corrected[2, 3]
        )

        qx, qy, qz, qw = (
            self.matrix_to_quaternion(
                t_corrected[
                    :3,
                    :3
                ]
            )
        )

        out.orientation.x = float(qx)
        out.orientation.y = float(qy)
        out.orientation.z = float(qz)
        out.orientation.w = float(qw)

        return out

    # ============================================================
    # POINT CLOUD CORRECTION
    # ============================================================

    @staticmethod
    def point_field_numpy_dtype(
        field,
        is_bigendian,
    ):
        """
        Return NumPy dtype for PointField FLOAT32/FLOAT64.
        """

        endian = (
            ">"
            if is_bigendian
            else "<"
        )

        if field.datatype == PointField.FLOAT32:
            return np.dtype(
                endian + "f4"
            )

        if field.datatype == PointField.FLOAT64:
            return np.dtype(
                endian + "f8"
            )

        return None

    def transform_cloud(
        self,
        msg,
    ):
        """
        Convert a FAST-LIO2 world-frame PointCloud2 to the corrected
        project coordinate system.

            p_corrected = C * p_native

        The PointCloud2 layout is handled row by row instead of assuming
        row_step == width * point_step. This is required because FAST-LIO2
        clouds may contain padding or layouts that are incompatible with a
        direct 2-D NumPy view.

        All PointCloud2 fields are preserved.
        """

        out = copy.deepcopy(msg)

        # The published cloud will be expressed in the common map frame.
        out.header.frame_id = self.fixed_frame

        if not self.apply_mount_correction:
            return out

        if not out.data:
            return out

        if out.point_step <= 0:
            self.get_logger().warning(
                "PointCloud2 point_step is invalid; "
                "publishing cloud without coordinate correction."
            )
            return out

        fields = {
            field.name: field
            for field in out.fields
        }

        # --------------------------------------------------------
        # Require XYZ
        # --------------------------------------------------------

        if not all(
            name in fields
            for name in ("x", "y", "z")
        ):
            self.get_logger().warning(
                "PointCloud2 does not contain x/y/z fields; "
                "publishing without coordinate correction."
            )
            return out

        # --------------------------------------------------------
        # Determine data types
        # --------------------------------------------------------

        dtype_x = self.point_field_numpy_dtype(
            fields["x"],
            out.is_bigendian,
        )

        dtype_y = self.point_field_numpy_dtype(
            fields["y"],
            out.is_bigendian,
        )

        dtype_z = self.point_field_numpy_dtype(
            fields["z"],
            out.is_bigendian,
        )

        if (
            dtype_x is None
            or dtype_y is None
            or dtype_z is None
        ):
            self.get_logger().warning(
                "Unsupported x/y/z PointCloud2 datatype; "
                "publishing without coordinate correction."
            )
            return out

        point_step = int(out.point_step)

        # Ensure XYZ fields actually fit inside one point.
        required_size = max(
            fields["x"].offset + dtype_x.itemsize,
            fields["y"].offset + dtype_y.itemsize,
            fields["z"].offset + dtype_z.itemsize,
        )

        if required_size > point_step:
            self.get_logger().warning(
                "Invalid PointCloud2 field layout; "
                "XYZ offsets exceed point_step. "
                "Publishing without coordinate correction."
            )
            return out

        # --------------------------------------------------------
        # Structured point dtype
        # --------------------------------------------------------

        point_dtype = np.dtype(
            {
                "names": [
                    "x",
                    "y",
                    "z",
                ],
                "formats": [
                    dtype_x,
                    dtype_y,
                    dtype_z,
                ],
                "offsets": [
                    int(fields["x"].offset),
                    int(fields["y"].offset),
                    int(fields["z"].offset),
                ],
                "itemsize": point_step,
            }
        )

        data = bytearray(out.data)

        rotation = self.mount_transform[
            :3,
            :3
        ]

        translation = self.mount_transform[
            :3,
            3
        ]

        width = int(out.width)
        height = max(
            1,
            int(out.height),
        )

        row_step = int(out.row_step)

        # Some PointCloud2 producers do not provide a useful row_step.
        if row_step <= 0:
            row_step = (
                width
                * point_step
            )

        # --------------------------------------------------------
        # Process row by row
        # --------------------------------------------------------

        for row in range(height):

            row_start = (
                row
                * row_step
            )

            if row_start >= len(data):
                break

            bytes_remaining = (
                len(data)
                - row_start
            )

            # Do not allow the NumPy view to extend beyond either:
            #
            #   - this PointCloud2 row
            #   - the actual buffer
            #
            row_bytes = min(
                row_step,
                bytes_remaining,
            )

            available_points = (
                row_bytes
                // point_step
            )

            point_count = min(
                width,
                available_points,
            )

            if point_count <= 0:
                continue

            points = np.ndarray(
                shape=(point_count,),
                dtype=point_dtype,
                buffer=data,
                offset=row_start,
                strides=(point_step,),
            )

            # Copy XYZ to float64 for transformation.
            xyz = np.column_stack(
                (
                    points["x"].astype(
                        np.float64,
                        copy=True,
                    ),
                    points["y"].astype(
                        np.float64,
                        copy=True,
                    ),
                    points["z"].astype(
                        np.float64,
                        copy=True,
                    ),
                )
            )

            transformed = (
                xyz
                @ rotation.T
                + translation
            )

            # Write only XYZ back into the original PointCloud2 buffer.
            # All other fields remain untouched.
            points["x"] = transformed[
                :,
                0
            ]

            points["y"] = transformed[
                :,
                1
            ]

            points["z"] = transformed[
                :,
                2
            ]

        out.data = bytes(data)

        return out

    # ============================================================
    # ODOMETRY
    # ============================================================

    def odom_callback(
        self,
        msg: Odometry,
    ):
        """
        FAST-LIO2 publishes its estimated body pose in its native world
        coordinate system.

        Convert that pose to the project's corrected Livox coordinate
        system before exposing it through /g1/slam/*.
        """

        self.odom_count += 1

        corrected_pose = (
            self.transform_pose(
                msg.pose.pose
            )
        )

        # --------------------------------------------------------
        # Common odometry
        # --------------------------------------------------------

        odom = Odometry()

        odom.header.stamp = (
            msg.header.stamp
        )

        odom.header.frame_id = (
            self.fixed_frame
        )

        odom.child_frame_id = (
            self.corrected_lidar_frame
        )

        # Preserve covariance and replace the actual pose.
        odom.pose = copy.deepcopy(
            msg.pose
        )

        odom.pose.pose = (
            corrected_pose
        )

        # Twist is currently preserved from FAST-LIO2.
        #
        # It is not used to construct the trajectory in this project.
        # The pose/path/cloud outputs are the important elements here.
        odom.twist = copy.deepcopy(
            msg.twist
        )

        self.odom_pub.publish(
            odom
        )

        # --------------------------------------------------------
        # Common PoseStamped
        # --------------------------------------------------------

        pose = PoseStamped()

        pose.header.stamp = (
            msg.header.stamp
        )

        pose.header.frame_id = (
            self.fixed_frame
        )

        pose.pose = copy.deepcopy(
            corrected_pose
        )

        self.pose_pub.publish(
            pose
        )

        self.last_pose = pose

        # --------------------------------------------------------
        # Dynamic TF
        # --------------------------------------------------------

        if self.publish_tf:
            self.publish_dynamic_tf(
                odom
            )

        # --------------------------------------------------------
        # Status
        # --------------------------------------------------------

        if self.publish_status_enabled:
            self.publish_status(
                odom,
                msg,
            )

    def publish_dynamic_tf(
        self,
        odom: Odometry,
    ):
        """
        Publish:

            map -> livox_corrected_frame

        livox_mount_tf_publisher.py separately publishes:

            livox_corrected_frame -> livox_frame
        """

        transform = (
            TransformStamped()
        )

        transform.header.stamp = (
            odom.header.stamp
        )

        transform.header.frame_id = (
            self.fixed_frame
        )

        transform.child_frame_id = (
            self.corrected_lidar_frame
        )

        transform.transform.translation.x = float(
            odom.pose.pose.position.x
        )

        transform.transform.translation.y = float(
            odom.pose.pose.position.y
        )

        transform.transform.translation.z = float(
            odom.pose.pose.position.z
        )

        transform.transform.rotation = (
            copy.deepcopy(
                odom.pose.pose.orientation
            )
        )

        self.tf_broadcaster.sendTransform(
            transform
        )

    # ============================================================
    # PATH
    # ============================================================

    def path_callback(
        self,
        msg: PathMsg,
    ):
        self.path_count += 1

        out = PathMsg()

        out.header.stamp = (
            msg.header.stamp
        )

        out.header.frame_id = (
            self.fixed_frame
        )

        poses = list(
            msg.poses
        )

        if (
            self.max_path_length > 0
            and len(poses)
            > self.max_path_length
        ):
            poses = poses[
                -self.max_path_length:
            ]

        for native_pose in poses:
            pose = PoseStamped()

            pose.header.stamp = (
                native_pose.header.stamp
            )

            pose.header.frame_id = (
                self.fixed_frame
            )

            pose.pose = (
                self.transform_pose(
                    native_pose.pose
                )
            )

            out.poses.append(
                pose
            )

        self.path_pub.publish(
            out
        )

    # ============================================================
    # POINT CLOUDS
    # ============================================================

    def aligned_callback(
        self,
        msg: PointCloud2,
    ):
        self.aligned_cloud_count += 1

        out = self.transform_cloud(
            msg
        )

        self.aligned_pub.publish(
            out
        )

    def map_callback(
        self,
        msg: PointCloud2,
    ):
        self.local_map_count += 1

        out = self.transform_cloud(
            msg
        )

        self.map_pub.publish(
            out
        )

    # ============================================================
    # STATUS
    # ============================================================

    def publish_status(
        self,
        odom: Odometry,
        native_odom: Odometry,
    ):
        position = (
            odom.pose.pose.position
        )

        orientation = (
            odom.pose.pose.orientation
        )

        native_position = (
            native_odom.pose.pose.position
        )

        status = {
            "algorithm": "fast_lio2",
            "accepted": True,
            "reason": "fast_lio2_odometry",
            "mount_correction_enabled": bool(
                self.apply_mount_correction
            ),
            "mount_rotation_deg": {
                "roll": float(
                    self.mount_rotation_deg[0]
                ),
                "pitch": float(
                    self.mount_rotation_deg[1]
                ),
                "yaw": float(
                    self.mount_rotation_deg[2]
                ),
            },
            "odom_messages": int(
                self.odom_count
            ),
            "path_messages": int(
                self.path_count
            ),
            "aligned_cloud_messages": int(
                self.aligned_cloud_count
            ),
            "local_map_messages": int(
                self.local_map_count
            ),
            "native_frame_id": (
                native_odom.header.frame_id
            ),
            "native_child_frame_id": (
                native_odom.child_frame_id
            ),
            "frame_id": (
                odom.header.frame_id
            ),
            "child_frame_id": (
                odom.child_frame_id
            ),
            "native_position": {
                "x": float(
                    native_position.x
                ),
                "y": float(
                    native_position.y
                ),
                "z": float(
                    native_position.z
                ),
            },
            "position": {
                "x": float(
                    position.x
                ),
                "y": float(
                    position.y
                ),
                "z": float(
                    position.z
                ),
            },
            "orientation": {
                "x": float(
                    orientation.x
                ),
                "y": float(
                    orientation.y
                ),
                "z": float(
                    orientation.z
                ),
                "w": float(
                    orientation.w
                ),
            },
        }

        out = String()

        out.data = json.dumps(
            status,
            separators=(
                ",",
                ":",
            ),
        )

        self.status_pub.publish(
            out
        )


def main(args=None):
    rclpy.init(
        args=args
    )

    node = FastLioOutputAdapter()

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