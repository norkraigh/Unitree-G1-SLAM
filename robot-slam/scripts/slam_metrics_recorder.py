import argparse
import bisect
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import rclpy
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from std_msgs.msg import String


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def percentile(values, p):
    if not values:
        return None
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    w = pos - lo
    return data[lo] * (1.0 - w) + data[hi] * w


def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config_metadata(config_file, fallback_algorithm):
    metadata = {
        "config_file": str(config_file) if config_file else None,
        "config_sha256": None,
        "optimization_enabled": False,
        "optimization_algorithm": "none",
        "execution_mode": "front_end_only",
        "algorithm_config_json": "{}",
        "global_optimization_config_json": "{}",
        "config_json": "{}",
    }
    if not config_file:
        return metadata

    path = Path(config_file).expanduser().resolve()
    if not path.exists():
        return metadata

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return metadata

    slam_cfg = config.get("slam", {})
    algorithm = str(slam_cfg.get("algorithm", fallback_algorithm)).strip().lower()
    global_cfg = config.get("global_optimization", {}) or {}
    opt_enabled = bool(global_cfg.get("enabled", False))
    opt_algorithm = str(global_cfg.get("algorithm", "pose_graph")).strip().lower() if opt_enabled else "none"

    if algorithm == "icp":
        # The custom ICP settings live under slam.algorithm_options.icp, while
        # mapping/motion/filter/performance policies are shared top-level.
        algorithm_cfg = {
            "icp": (
                slam_cfg.get("algorithm_options", {}).get("icp", {}) or {}
            ),
            "mapping": config.get("mapping", {}),
            "motion_model": config.get("motion_model", {}),
            "filters": config.get("filters", {}),
            "performance": config.get("performance", {}),
        }
    else:
        algorithm_cfg = (
            slam_cfg.get("algorithm_options", {}).get(algorithm, {}) or {}
        )

    metadata.update({
        "config_file": str(path),
        "config_sha256": file_sha256(path),
        "optimization_enabled": opt_enabled,
        "optimization_algorithm": opt_algorithm,
        "execution_mode": (
            f"{algorithm}+{opt_algorithm}" if opt_enabled else f"{algorithm}_only"
        ),
        "algorithm_config_json": compact_json(algorithm_cfg),
        "global_optimization_config_json": compact_json(global_cfg),
        "config_json": compact_json(config),
    })
    return metadata


class SlamMetricsRecorder(Node):
    def __init__(self, args):
        super().__init__("slam_metrics_recorder")
        self.args = args
        self.poses = []
        self.optimized_poses = []
        self.status_rows = []
        self.config_metadata = load_config_metadata(args.config_file, args.algorithm)

        self.create_subscription(Odometry, args.odom_topic, self.odom_cb, 100)
        self.create_subscription(String, args.status_topic, self.status_cb, 100)
        self.create_subscription(
            String, args.optimized_status_topic, self.optimized_status_cb, 100
        )
        self.create_subscription(
            PathMsg, args.optimized_path_topic, self.optimized_path_cb, 10
        )

        self.get_logger().info(
            f"Recording metrics: odom={args.odom_topic}, status={args.status_topic}, "
            f"optimized_path={args.optimized_path_topic}"
        )

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.poses.append({
            "stamp_s": stamp_to_sec(msg.header.stamp),
            "x_m": float(p.x),
            "y_m": float(p.y),
            "z_m": float(p.z),
            "yaw_rad": float(yaw),
            "yaw_deg": math.degrees(yaw),
        })

    def _append_status(self, msg, source):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, dict):
            return

        row = {
            "status_source": source,
            "stamp_s": (
                float(data.get("stamp_sec", 0))
                + float(data.get("stamp_nanosec", 0)) * 1e-9
                if "stamp_sec" in data else None
            ),
            "accepted": data.get("accepted"),
            "reason": data.get("reason"),
            "points": data.get("points"),
            "fitness": data.get("fitness"),
            "rmse": data.get("rmse"),
            "processing_ms": data.get("processing_ms"),
            "profile_preprocess_ms": data.get("profile_preprocess_ms"),
            "profile_deskew_ms": data.get("profile_deskew_ms"),
            "profile_icp_ms": data.get("profile_icp_ms"),
            "profile_icp_calls": data.get("profile_icp_calls"),
            "estimated_dropped_scans": data.get(
                "estimated_dropped_scans"
            ),
            "pending_lidar_queue_drops": data.get(
                "pending_lidar_queue_drops"
            ),
            "preprocess_ms": data.get("preprocess_ms"),
            "icp_ms": data.get("icp_ms"),
            "primary_icp_ms": data.get("primary_icp_ms"),
            "prediction_icp_ms": data.get("prediction_icp_ms"),
            "recovery_icp_ms": data.get("recovery_icp_ms"),
            "recovery_prepare_ms": data.get("recovery_prepare_ms"),
            "mapping_ms": data.get("mapping_ms"),
            "publish_ms": data.get("publish_ms"),
            "other_ms": data.get("other_ms"),
            "icp_calls": data.get("icp_calls"),
            "scan_dt_sec": data.get("scan_dt_sec"),
            "dropped_scans_estimated_last": data.get("dropped_scans_estimated_last"),
            "dropped_scans_estimated_total": data.get("dropped_scans_estimated_total"),
            "icp_target": data.get("icp_target"),
            "keyframe_added": data.get("keyframe_added"),
            "local_map_frames": data.get("local_map_frames"),
            "local_map_points": data.get("local_map_points"),
            "consecutive_rejections": data.get("consecutive_rejections"),
            "recovery_active": data.get("recovery_active"),
            "recovery_attempted": data.get("recovery_attempted"),
            "recovery_success": data.get("recovery_success"),
            "recovery_provisional": data.get("recovery_provisional"),
            "recovery_confirmed": data.get("recovery_confirmed"),
            "recovery_confirmations": data.get("recovery_confirmations"),
            "recovery_reason": data.get("recovery_reason"),
            "recovery_seed_yaw_offset_deg": data.get("recovery_seed_yaw_offset_deg"),
            "recovery_imu_error_deg": data.get("recovery_imu_error_deg"),
            "pose_graph_nodes": data.get("pose_graph_nodes", data.get("keyframes")),
            "pose_graph_edges": data.get("pose_graph_edges", data.get("edges")),
            "loop_closures": data.get("loop_closures"),
            "loop_detection_ms": data.get("loop_detection_ms"),
            "optimization_ms": data.get("optimization_ms"),
            "map_publish_ms": data.get("map_publish_ms"),
            "backend": data.get("backend"),
            "optimized": data.get("optimized"),
        }
        self.status_rows.append(row)

    def status_cb(self, msg):
        self._append_status(msg, "front_end")

    def optimized_status_cb(self, msg):
        self._append_status(msg, "optimized_backend")

    def optimized_path_cb(self, msg):
        poses = []
        for pose_stamped in msg.poses:
            p = pose_stamped.pose.position
            yaw = quaternion_to_yaw(pose_stamped.pose.orientation)
            poses.append({
                "stamp_s": stamp_to_sec(pose_stamped.header.stamp),
                "x_m": float(p.x),
                "y_m": float(p.y),
                "z_m": float(p.z),
                "yaw_rad": float(yaw),
                "yaw_deg": math.degrees(yaw),
            })
        if poses:
            # The backend republishes the complete optimized path. Keep only
            # the newest complete version instead of concatenating duplicates.
            self.optimized_poses = poses

    def _prepare_monotonic_poses(self, poses):
        """Sort by timestamp and keep the newest sample for duplicate stamps."""
        by_stamp = {}
        for pose in poses:
            t = safe_float(pose.get("stamp_s"))
            if t is None:
                continue
            by_stamp[t] = pose
        return [by_stamp[t] for t in sorted(by_stamp)]

    def _interpolate_raw_pose(self, raw_poses, raw_times, target_t):
        """Interpolate the raw odometry at target_t.

        The optimized path is sparse (pose-graph keyframes), while raw odometry is
        dense.  Interpolating raw odometry at the optimized keyframe timestamps
        makes trajectory metrics comparable without changing the optimized path.
        """
        if not raw_poses or target_t < raw_times[0] or target_t > raw_times[-1]:
            return None

        idx = bisect.bisect_left(raw_times, target_t)
        if idx < len(raw_times) and abs(raw_times[idx] - target_t) <= 1e-9:
            src = raw_poses[idx]
            return dict(src)

        if idx == 0 or idx >= len(raw_times):
            return None

        a = raw_poses[idx - 1]
        b = raw_poses[idx]
        ta = raw_times[idx - 1]
        tb = raw_times[idx]
        if tb <= ta:
            return None

        # Reject interpolation across a long raw-odometry gap. This prevents a
        # missing-data section from being treated as a valid comparable segment.
        if (tb - ta) > self.args.alignment_max_gap_s:
            return None

        alpha = (target_t - ta) / (tb - ta)
        yaw_delta = angle_diff(b["yaw_rad"], a["yaw_rad"])
        yaw = a["yaw_rad"] + alpha * yaw_delta
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        return {
            "stamp_s": float(target_t),
            "x_m": a["x_m"] + alpha * (b["x_m"] - a["x_m"]),
            "y_m": a["y_m"] + alpha * (b["y_m"] - a["y_m"]),
            "z_m": a["z_m"] + alpha * (b["z_m"] - a["z_m"]),
            "yaw_rad": yaw,
            "yaw_deg": math.degrees(yaw),
        }

    def _build_comparable_trajectories(self):
        """Return RAW and optimized trajectories at identical timestamps."""
        raw = self._prepare_monotonic_poses(self.poses)
        optimized = self._prepare_monotonic_poses(self.optimized_poses)
        if len(raw) < 2 or len(optimized) < 2:
            return [], []

        raw_times = [p["stamp_s"] for p in raw]
        raw_comparable = []
        optimized_comparable = []

        for opt_pose in optimized:
            t = opt_pose["stamp_s"]
            raw_pose = self._interpolate_raw_pose(raw, raw_times, t)
            if raw_pose is None:
                continue
            raw_comparable.append(raw_pose)
            optimized_comparable.append(opt_pose)

        return raw_comparable, optimized_comparable

    def _trajectory_metrics(self, poses):
        metrics = {"pose_samples": len(poses)}
        if not poses:
            return metrics

        first = poses[0]
        last = poses[-1]
        duration = max(0.0, last["stamp_s"] - first["stamp_s"])

        step_2d = []
        step_3d = []
        step_yaw_deg = []
        dt_values = []
        jump_translation_count = 0
        jump_yaw_count = 0

        for a, b in zip(poses, poses[1:]):
            dx = b["x_m"] - a["x_m"]
            dy = b["y_m"] - a["y_m"]
            dz = b["z_m"] - a["z_m"]
            d2 = math.hypot(dx, dy)
            d3 = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = abs(math.degrees(angle_diff(b["yaw_rad"], a["yaw_rad"])))
            dt = b["stamp_s"] - a["stamp_s"]

            step_2d.append(d2)
            step_3d.append(d3)
            step_yaw_deg.append(dyaw)
            if dt > 0:
                dt_values.append(dt)
            if d2 > self.args.jump_translation_m:
                jump_translation_count += 1
            if dyaw > self.args.jump_yaw_deg:
                jump_yaw_count += 1

        dx = last["x_m"] - first["x_m"]
        dy = last["y_m"] - first["y_m"]
        dz = last["z_m"] - first["z_m"]

        metrics.update({
            "slam_duration_s": duration,
            "odom_mean_hz": ((len(poses) - 1) / duration) if duration > 0 and len(poses) > 1 else None,
            "trajectory_length_2d_m": sum(step_2d),
            "trajectory_length_3d_m": sum(step_3d),
            "start_end_distance_2d_m": math.hypot(dx, dy),
            "start_end_distance_3d_m": math.sqrt(dx * dx + dy * dy + dz * dz),
            "start_end_yaw_error_deg": abs(math.degrees(angle_diff(last["yaw_rad"], first["yaw_rad"]))),
            "max_step_translation_m": max(step_2d) if step_2d else 0.0,
            "p95_step_translation_m": percentile(step_2d, 0.95),
            "max_step_yaw_deg": max(step_yaw_deg) if step_yaw_deg else 0.0,
            "p95_step_yaw_deg": percentile(step_yaw_deg, 0.95),
            "translation_jump_threshold_m": self.args.jump_translation_m,
            "translation_jump_count": jump_translation_count,
            "yaw_jump_threshold_deg": self.args.jump_yaw_deg,
            "yaw_jump_count": jump_yaw_count,
            "mean_pose_dt_s": mean(dt_values) if dt_values else None,
            "final_x_m": last["x_m"],
            "final_y_m": last["y_m"],
            "final_z_m": last["z_m"],
            "final_yaw_deg": last["yaw_deg"],
        })
        return metrics

    def _status_metrics(self):
        if not self.status_rows:
            return {}

        front = [
            r for r in self.status_rows
            if r.get("status_source") == "front_end"
        ]
        backend = [
            r for r in self.status_rows
            if r.get("status_source") == "optimized_backend"
        ]

        accepted_values = [
            r["accepted"] for r in front
            if isinstance(r.get("accepted"), bool)
        ]
        accepted_count = sum(1 for v in accepted_values if v)
        rejected_count = sum(1 for v in accepted_values if not v)

        def vals(rows, key):
            return [
                v for r in rows
                if (v := safe_float(r.get(key))) is not None
            ]

        fitness = vals(front, "fitness")
        rmse = vals(front, "rmse")
        proc = vals(front, "processing_ms")
        points = vals(front, "points")

        # New ICP front-end profiling fields. Keep the previous field names as
        # fallbacks so old benchmark runs remain readable with this recorder.
        profile_preprocess = vals(front, "profile_preprocess_ms")
        legacy_preprocess = vals(front, "preprocess_ms")
        preprocess = (
            profile_preprocess
            if profile_preprocess
            else legacy_preprocess
        )

        deskew = vals(front, "profile_deskew_ms")

        profile_icp = vals(front, "profile_icp_ms")
        legacy_icp = vals(front, "icp_ms")
        icp = profile_icp if profile_icp else legacy_icp

        profile_icp_calls = vals(front, "profile_icp_calls")

        primary_icp = vals(front, "primary_icp_ms")
        prediction_icp = vals(front, "prediction_icp_ms")
        recovery_icp = vals(front, "recovery_icp_ms")
        recovery_prepare = vals(front, "recovery_prepare_ms")
        mapping = vals(front, "mapping_ms")
        publish = vals(front, "publish_ms")
        other = vals(front, "other_ms")
        scan_dt = vals(front, "scan_dt_sec")
        consecutive = vals(front, "consecutive_rejections")

        backend_proc = vals(backend, "processing_ms")
        backend_loop = vals(backend, "loop_detection_ms")
        backend_opt = [v for v in vals(backend, "optimization_ms") if v > 0.0]
        backend_map = [v for v in vals(backend, "map_publish_ms") if v > 0.0]

        def mean_or_none(v):
            return mean(v) if v else None

        def p95_or_none(v):
            return percentile(v, 0.95) if v else None

        # The current ICP node publishes cumulative counters using the names
        # below.  Fall back to the older recorder/node names for compatibility.
        latest_dropped = next(
            (
                r.get("estimated_dropped_scans")
                for r in reversed(front)
                if r.get("estimated_dropped_scans") is not None
            ),
            None,
        )
        if latest_dropped is None:
            latest_dropped = next(
                (
                    r.get("dropped_scans_estimated_total")
                    for r in reversed(front)
                    if r.get("dropped_scans_estimated_total") is not None
                ),
                None,
            )

        latest_queue_drops = next(
            (
                r.get("pending_lidar_queue_drops")
                for r in reversed(front)
                if r.get("pending_lidar_queue_drops") is not None
            ),
            None,
        )

        return {
            "accepted_frames": accepted_count if accepted_values else None,
            "rejected_frames": rejected_count if accepted_values else None,
            "acceptance_rate_pct": (
                100.0 * accepted_count / len(accepted_values)
                if accepted_values else None
            ),
            "fitness_mean": mean_or_none(fitness),
            "fitness_median": median(fitness) if fitness else None,
            "fitness_p05": percentile(fitness, 0.05),
            "fitness_min": min(fitness) if fitness else None,
            "rmse_mean_m": mean_or_none(rmse),
            "rmse_p95_m": p95_or_none(rmse),
            "rmse_max_m": max(rmse) if rmse else None,
            "processing_mean_ms": mean_or_none(proc),
            "processing_p95_ms": p95_or_none(proc),
            "processing_max_ms": max(proc) if proc else None,
            "processing_over_100ms_count": sum(1 for v in proc if v > 100.0),
            "processing_over_150ms_count": sum(1 for v in proc if v > 150.0),
            "preprocess_mean_ms": mean_or_none(preprocess),
            "preprocess_p95_ms": p95_or_none(preprocess),
            "deskew_mean_ms": mean_or_none(deskew),
            "deskew_p95_ms": p95_or_none(deskew),
            "icp_mean_ms": mean_or_none(icp),
            "icp_p95_ms": p95_or_none(icp),
            "icp_calls_mean": mean_or_none(profile_icp_calls),
            "primary_icp_mean_ms": mean_or_none([v for v in primary_icp if v > 0.0]),
            "primary_icp_p95_ms": p95_or_none([v for v in primary_icp if v > 0.0]),
            "prediction_icp_mean_ms": mean_or_none([v for v in prediction_icp if v > 0.0]),
            "recovery_icp_mean_ms": mean_or_none([v for v in recovery_icp if v > 0.0]),
            "recovery_icp_p95_ms": p95_or_none([v for v in recovery_icp if v > 0.0]),
            "recovery_prepare_mean_ms": mean_or_none([v for v in recovery_prepare if v > 0.0]),
            "mapping_mean_ms": mean_or_none(mapping),
            "publish_mean_ms": mean_or_none(publish),
            "other_mean_ms": mean_or_none(other),
            "points_mean": mean_or_none(points),
            "scan_dt_mean_s": mean_or_none(scan_dt),
            "scan_dt_p95_s": p95_or_none(scan_dt),
            "dropped_scans_estimated_total": (
                int(float(latest_dropped))
                if safe_float(latest_dropped) is not None
                else None
            ),
            "pending_lidar_queue_drops_total": (
                int(float(latest_queue_drops))
                if safe_float(latest_queue_drops) is not None
                else None
            ),
            "max_consecutive_rejections": int(max(consecutive)) if consecutive else None,
            "recovery_attempts": sum(1 for r in front if r.get("recovery_attempted") is True),
            "recovery_provisional_candidates": sum(1 for r in front if r.get("recovery_provisional") is True),
            "recovery_confirmed_successes": sum(1 for r in front if r.get("recovery_success") is True),
            "recovery_imu_rejections": sum(
                1 for r in front
                if r.get("recovery_reason") == "recovery_imu_rotation_mismatch"
            ),
            "keyframes_added": sum(1 for r in front if r.get("keyframe_added") is True),
            "local_map_points_final": next((r.get("local_map_points") for r in reversed(front) if r.get("local_map_points") is not None), None),
            "pose_graph_nodes_final": next((r.get("pose_graph_nodes") for r in reversed(backend) if r.get("pose_graph_nodes") is not None), None),
            "pose_graph_edges_final": next((r.get("pose_graph_edges") for r in reversed(backend) if r.get("pose_graph_edges") is not None), None),
            "loop_closures_final": next((r.get("loop_closures") for r in reversed(backend) if r.get("loop_closures") is not None), None),
            "backend_processing_mean_ms": mean_or_none(backend_proc),
            "backend_processing_p95_ms": p95_or_none(backend_proc),
            "backend_processing_max_ms": max(backend_proc) if backend_proc else None,
            "backend_loop_detection_mean_ms": mean_or_none(backend_loop),
            "backend_loop_detection_p95_ms": p95_or_none(backend_loop),
            "backend_optimization_mean_ms": mean_or_none(backend_opt),
            "backend_optimization_max_ms": max(backend_opt) if backend_opt else None,
            "backend_map_publish_mean_ms": mean_or_none(backend_map),
            "backend_map_publish_max_ms": max(backend_map) if backend_map else None,
        }

    def _base_summary(self):
        opt_enabled = bool(self.config_metadata.get("optimization_enabled", False))
        return {
            "run_id": self.args.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "bag": self.args.bag_name,
            "algorithm": self.args.algorithm,
            "optimization_enabled": opt_enabled,
            "optimization_algorithm": self.config_metadata.get("optimization_algorithm", "none"),
            "execution_mode": self.config_metadata.get("execution_mode"),
            "optimized_path_available": len(self.optimized_poses) >= 2,
            "variant": self.args.variant,
            "config_file": self.config_metadata.get("config_file"),
            "config_sha256": self.config_metadata.get("config_sha256"),
            "algorithm_config_json": self.config_metadata.get("algorithm_config_json"),
            "global_optimization_config_json": self.config_metadata.get("global_optimization_config_json"),
            "config_json": self.config_metadata.get("config_json"),
            "odom_topic": self.args.odom_topic,
            "status_topic": self.args.status_topic,
            "optimized_status_topic": self.args.optimized_status_topic,
            "optimized_path_topic": self.args.optimized_path_topic,
            "raw_odom_samples": len(self.poses),
            "optimized_path_samples": len(self.optimized_poses),
            "status_samples": len(self.status_rows),
        }

    def build_summaries(self):
        """Return comparable RAW/OPTIMIZED rows when a pose graph path exists.

        Primary trajectory metrics in the two rows are computed at exactly the
        same timestamps. Dense raw-odometry metrics are retained in prefixed
        columns so output frequency/performance information is not lost.
        """
        base = self._base_summary()
        status_metrics = self._status_metrics()
        summaries = []

        raw_full_metrics = self._trajectory_metrics(self._prepare_monotonic_poses(self.poses))
        raw_full_prefixed = {f"raw_full_{k}": v for k, v in raw_full_metrics.items()}

        raw_comparable, optimized_comparable = self._build_comparable_trajectories()
        has_comparable_pair = len(raw_comparable) >= 2 and len(optimized_comparable) >= 2

        common = dict(base)
        common.update(raw_full_prefixed)
        common.update({
            "trajectory_comparison_aligned": has_comparable_pair,
            "alignment_method": "raw_linear_interpolation_at_optimized_timestamps" if has_comparable_pair else "none",
            "alignment_max_gap_s": self.args.alignment_max_gap_s,
            "comparison_pose_samples": len(raw_comparable) if has_comparable_pair else None,
            "comparison_start_stamp_s": raw_comparable[0]["stamp_s"] if has_comparable_pair else None,
            "comparison_end_stamp_s": raw_comparable[-1]["stamp_s"] if has_comparable_pair else None,
        })

        raw = dict(common)
        raw.update({
            "comparison_method": f"{self.args.algorithm}_raw",
            "result_mode": "raw",
            "optimization_applied_to_result": False,
            "trajectory_metrics_mode": "raw_comparable" if has_comparable_pair else "raw_full",
            "trajectory_metrics_source": (
                f"{self.args.odom_topic} interpolated at {self.args.optimized_path_topic} timestamps"
                if has_comparable_pair else self.args.odom_topic
            ),
        })
        raw.update(self._trajectory_metrics(raw_comparable if has_comparable_pair else self.poses))
        raw.update(status_metrics)
        summaries.append(raw)

        if has_comparable_pair:
            optimized = dict(common)
            optimized.update({
                "comparison_method": f"{self.args.algorithm}_optimized",
                "result_mode": "optimized",
                "optimization_applied_to_result": True,
                "trajectory_metrics_mode": "optimized_comparable",
                "trajectory_metrics_source": self.args.optimized_path_topic,
            })
            optimized.update(self._trajectory_metrics(optimized_comparable))
            optimized.update(status_metrics)
            summaries.append(optimized)

        return summaries

    def save(self):
        output_dir = Path(self.args.output_dir).expanduser().resolve()
        run_dir = output_dir / "runs" / self.args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        write_csv(run_dir / "trajectory.csv", self.poses)
        write_csv(run_dir / "optimized_trajectory.csv", self.optimized_poses)
        raw_comparable, optimized_comparable = self._build_comparable_trajectories()
        write_csv(run_dir / "raw_comparable_trajectory.csv", raw_comparable)
        write_csv(run_dir / "optimized_comparable_trajectory.csv", optimized_comparable)
        write_csv(run_dir / "status.csv", self.status_rows)

        summaries = self.build_summaries()
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"results": summaries}, f, indent=2, ensure_ascii=False)

        # Convenience files: each row is also available independently.
        if summaries:
            with (run_dir / "summary_raw.json").open("w", encoding="utf-8") as f:
                json.dump(summaries[0], f, indent=2, ensure_ascii=False)
        if len(summaries) > 1:
            with (run_dir / "summary_optimized.json").open("w", encoding="utf-8") as f:
                json.dump(summaries[1], f, indent=2, ensure_ascii=False)

        master = output_dir / "tfm_slam_benchmark.csv"
        exists = master.exists()
        fieldnames = []
        for summary in summaries:
            for key in summary.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        # Keep a stable master schema. If new fields appear later, rewrite the
        # old file with the union of columns instead of silently dropping data.
        old_rows = []
        old_fields = []
        if exists:
            with master.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                old_rows = list(reader)

        preferred_fields = [
            "run_id", "timestamp", "bag",
            "algorithm", "comparison_method", "result_mode",
            "optimization_enabled", "optimization_algorithm",
            "optimization_applied_to_result", "execution_mode",
            "trajectory_metrics_mode", "trajectory_metrics_source",
            "trajectory_comparison_aligned", "alignment_method",
            "alignment_max_gap_s", "comparison_pose_samples",
            "comparison_start_stamp_s", "comparison_end_stamp_s",
            "optimized_path_available", "variant",
            "config_file", "config_sha256", "algorithm_config_json",
            "global_optimization_config_json", "config_json",
        ]
        all_fields = old_fields + [k for k in fieldnames if k not in old_fields]
        merged_fields = [k for k in preferred_fields if k in all_fields]
        merged_fields += [k for k in all_fields if k not in merged_fields]
        if not merged_fields:
            merged_fields = fieldnames

        with master.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
            writer.writeheader()
            for row in old_rows:
                writer.writerow(row)
            for summary in summaries:
                writer.writerow(summary)

        modes = ", ".join(row.get("result_mode", "unknown") for row in summaries)
        self.get_logger().info(
            f"Metrics written to: {master} ({len(summaries)} row(s): {modes})"
        )
        self.get_logger().info(f"Run details written to: {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--bag-name", required=True)
    parser.add_argument("--variant", default="")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--odom-topic", default="/g1/slam/odom")
    parser.add_argument("--status-topic", default="/g1/slam/status")
    parser.add_argument("--optimized-status-topic", default="/g1/slam/optimized/status")
    parser.add_argument("--optimized-path-topic", default="/g1/slam/optimized/path")
    parser.add_argument("--jump-translation-m", type=float, default=0.75)
    parser.add_argument("--jump-yaw-deg", type=float, default=30.0)
    parser.add_argument(
        "--alignment-max-gap-s",
        type=float,
        default=0.5,
        help="Maximum raw odometry gap allowed when interpolating at optimized timestamps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SlamMetricsRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
