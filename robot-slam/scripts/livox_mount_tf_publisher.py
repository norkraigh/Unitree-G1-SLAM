import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


class LivoxMountTfPublisher(Node):
    """
    Publishes the static transform that represents the physical mounting
    correction of the Livox MID-360.

    The transform is defined centrally in config/livox_slam_config.json:

        sensors.livox.translation_m
        sensors.livox.rotation_deg

    TF convention used by the project:

        corrected_lidar_frame -> robot_frame

    With the current G1 mounting this is normally:

        livox_corrected_frame -> livox_frame
        roll = 180 deg

    The node intentionally does not modify /livox/lidar. The original
    PointCloud2 message, including per-point timestamps and Livox-specific
    fields, remains untouched.
    """

    def __init__(self):
        super().__init__("livox_mount_tf_publisher")

        self.declare_parameter("config_file", "")

        config_file = str(
            self.get_parameter("config_file").value
        ).strip()

        if not config_file:
            raise RuntimeError(
                "Parameter 'config_file' is required."
            )

        config_path = Path(config_file).expanduser().resolve()

        if not config_path.exists():
            raise RuntimeError(
                f"Configuration file not found: {config_path}"
            )

        self.config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        frames = self.config.get("frames", {})
        livox_cfg = (
            self.config
            .get("sensors", {})
            .get("livox", {})
        )

        self.raw_frame = str(
            frames.get("robot_frame", "livox_frame")
        )

        self.corrected_frame = str(
            frames.get(
                "corrected_lidar_frame",
                "livox_corrected_frame",
            )
        )

        self.enabled = bool(
            livox_cfg.get("apply_mount_correction", False)
        )

        translation_cfg = livox_cfg.get(
            "translation_m",
            {},
        )

        rotation_cfg = livox_cfg.get(
            "rotation_deg",
            {},
        )

        self.translation = (
            float(translation_cfg.get("x", 0.0)),
            float(translation_cfg.get("y", 0.0)),
            float(translation_cfg.get("z", 0.0)),
        )

        self.rotation_deg = (
            float(rotation_cfg.get("roll", 0.0)),
            float(rotation_cfg.get("pitch", 0.0)),
            float(rotation_cfg.get("yaw", 0.0)),
        )

        self.broadcaster = StaticTransformBroadcaster(self)

        if self.enabled:
            self.publish_transform()

            self.get_logger().info(
                "Livox mount correction TF published: "
                f"{self.corrected_frame} -> {self.raw_frame}; "
                f"translation={self.translation}; "
                f"rotation_deg={self.rotation_deg}"
            )
        else:
            self.get_logger().info(
                "Livox mount correction disabled in configuration. "
                "No static correction TF was published."
            )

    @staticmethod
    def quaternion_from_rpy(roll, pitch, yaw):
        """Return quaternion (x, y, z, w) from roll/pitch/yaw radians."""

        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        norm = math.sqrt(
            qx * qx
            + qy * qy
            + qz * qz
            + qw * qw
        )

        if norm < 1e-12:
            return 0.0, 0.0, 0.0, 1.0

        return (
            qx / norm,
            qy / norm,
            qz / norm,
            qw / norm,
        )

    def publish_transform(self):
        roll, pitch, yaw = [
            math.radians(value)
            for value in self.rotation_deg
        ]

        qx, qy, qz, qw = self.quaternion_from_rpy(
            roll,
            pitch,
            yaw,
        )

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.corrected_frame
        transform.child_frame_id = self.raw_frame

        transform.transform.translation.x = self.translation[0]
        transform.transform.translation.y = self.translation[1]
        transform.transform.translation.z = self.translation[2]

        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init()

    node = LivoxMountTfPublisher()

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