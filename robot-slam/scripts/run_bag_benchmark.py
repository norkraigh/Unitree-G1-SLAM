import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime


DEFAULT_PLAYBACK_TOPICS = [
    "/livox/lidar",
    "/livox/imu",
    "/camera/color/image_raw",
    "/camera/color/camera_info",
]


# ============================================================
# STATUS JSONL LOGGER
# ============================================================

STATUS_LOGGER_CODE = r"""
import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


topic = sys.argv[1]
output_path = Path(sys.argv[2])

output_path.parent.mkdir(parents=True, exist_ok=True)


class StatusJsonlLogger(Node):
    def __init__(self):
        super().__init__("tfm_status_jsonl_logger")

        self.output_file = output_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )

        self.subscription = self.create_subscription(
            String,
            topic,
            self.status_callback,
            100,
        )

    def status_callback(self, msg):
        try:
            parsed = json.loads(msg.data)

            if isinstance(parsed, dict):
                record = parsed
            else:
                record = {
                    "_parsed_status": parsed,
                }

        except Exception as exc:
            record = {
                "_parse_error": True,
                "_parse_error_message": str(exc),
                "_raw": msg.data,
            }

        self.output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

        # Flush every status message so an interrupted benchmark
        # still leaves an analyzable file.
        self.output_file.flush()

    def close_file(self):
        try:
            self.output_file.flush()
            self.output_file.close()
        except Exception:
            pass


def main():
    rclpy.init()

    node = StatusJsonlLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_file()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
"""


def terminate_group(proc, sig=signal.SIGINT, timeout=5.0):
    if proc is None:
        return

    # start_group() creates a new process group whose PGID equals the launcher
    # PID. run_slam.py may exit after spawning ICP/KISS-ICP/FAST-LIO2, while
    # its children remain alive in the same group.
    pgid = proc.pid

    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return

    if proc.poll() is None:
        try:
            proc.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return

            try:
                proc.wait(timeout=2.0)

            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def start_group(cmd, cwd):
    print(
        "+",
        " ".join(str(x) for x in cmd),
        flush=True,
    )

    return subprocess.Popen(
        cmd,
        cwd=cwd,
        preexec_fn=os.setsid,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one rosbag through the selected SLAM algorithm "
            "and save TFM metrics."
        )
    )

    parser.add_argument("bag")

    parser.add_argument(
        "--config",
        default="config/livox_slam_config.json",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
    )

    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_PLAYBACK_TOPICS,
        help=(
            "Topics to replay from the bag. By default only the LiDAR, "
            "Livox IMU, RGB image and RGB camera_info are published."
        ),
    )

    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Replay every topic stored in the bag.",
    )

    parser.add_argument(
        "--drain-delay",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--jump-translation-m",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--jump-yaw-deg",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--alignment-max-gap-s",
        type=float,
        default=0.5,
        help=(
            "Maximum raw odometry gap allowed for RAW/optimized "
            "timestamp alignment."
        ),
    )

    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help=(
            "Keep SLAM nodes alive after rosbag playback finishes "
            "so the final state can be inspected in Foxglove. "
            "Press Ctrl+C to stop."
        ),
    )

    args = parser.parse_args()

    # ============================================================
    # PATHS
    # ============================================================

    script_path = Path(__file__).resolve()

    root = (
        script_path.parent.parent
        if script_path.parent.name == "scripts"
        else script_path.parent
    )

    config_path = (
        (root / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config).resolve()
    )

    bag_path = Path(
        args.bag
    ).expanduser().resolve()

    if not config_path.exists():
        raise SystemExit(
            f"Config file not found: {config_path}"
        )

    if not bag_path.exists():
        raise SystemExit(
            f"Bag not found: {bag_path}"
        )

    # ============================================================
    # CONFIG
    # ============================================================

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    algorithm = str(
        config
        .get("slam", {})
        .get("algorithm", "icp")
    ).strip().lower()

    global_opt_cfg = (
        config.get(
            "global_optimization",
            {},
        )
        or {}
    )

    global_opt_enabled = bool(
        global_opt_cfg.get(
            "enabled",
            False,
        )
    )

    global_opt_algorithm = (
        str(
            global_opt_cfg.get(
                "algorithm",
                "pose_graph",
            )
        ).strip().lower()
        if global_opt_enabled
        else "none"
    )

    variant = (
        f"{algorithm}_{global_opt_algorithm}"
        if global_opt_enabled
        else algorithm
    )

    mode_label = (
        f"{algorithm}+{global_opt_algorithm}"
        if global_opt_enabled
        else f"{algorithm}_only"
    )

    # ============================================================
    # RUN ID
    # ============================================================

    run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{variant}_{bag_path.name}"
    )

    safe_run_id = "".join(
        c
        if c.isalnum() or c in "-_."
        else "_"
        for c in run_id
    )

    run_dir = (
        root
        / args.output_dir
        / "runs"
        / safe_run_id
    ).resolve()

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_jsonl_path = (
        run_dir
        / "slam_status.jsonl"
    )

    config_copy_path = (
        run_dir
        / "livox_slam_config.json"
    )

    metadata_path = (
        run_dir
        / "run_metadata.json"
    )

    # Save the exact configuration before starting the test.
    config_copy_path.write_text(
        config_path.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    # ============================================================
    # RUN METADATA
    # ============================================================

    metadata = {
        "run_id": safe_run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "bag": str(bag_path),
        "bag_name": bag_path.name,
        "algorithm": algorithm,
        "global_optimization_enabled": global_opt_enabled,
        "global_optimization_algorithm": global_opt_algorithm,
        "variant": variant,
        "execution_mode": mode_label,
        "rate": float(args.rate),
        "all_topics": bool(args.all_topics),
        "topics": (
            "ALL"
            if args.all_topics
            else list(args.topics)
        ),
        "config_file": str(config_path),
        "status_topic": "/g1/slam/status",
        "status_log": str(status_jsonl_path),
        "alignment_max_gap_s": float(
            args.alignment_max_gap_s
        ),
        "jump_translation_m": float(
            args.jump_translation_m
        ),
        "jump_yaw_deg": float(
            args.jump_yaw_deg
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # PROJECT FILES
    # ============================================================

    run_slam = (
        root
        / "run_slam.py"
    )

    recorder = (
        root
        / "scripts"
        / "slam_metrics_recorder.py"
    )

    if not run_slam.exists():
        raise SystemExit(
            f"run_slam.py not found: {run_slam}"
        )

    if not recorder.exists():
        raise SystemExit(
            f"Metrics recorder not found: {recorder}"
        )

    # All algorithms are normalized to the common project interface.
    odom_topic = "/g1/slam/odom"
    status_topic = "/g1/slam/status"

    # ============================================================
    # PRINT CONFIGURATION
    # ============================================================

    print("=" * 60)
    print("TFM benchmark configuration")
    print(f"Algorithm: {algorithm}")
    print(
        f"Optimization enabled: "
        f"{global_opt_enabled}"
    )
    print(
        f"Optimization algorithm: "
        f"{global_opt_algorithm}"
    )
    print(
        f"Execution mode: "
        f"{mode_label}"
    )
    print(
        f"Config file: "
        f"{config_path}"
    )
    print(
        f"Playback rate: "
        f"{args.rate}"
    )

    if args.all_topics:
        print(
            "Bag playback: ALL recorded topics"
        )
    else:
        print(
            "Bag playback topics: "
            + ", ".join(args.topics)
        )

    print(
        f"RAW/optimized alignment max gap: "
        f"{args.alignment_max_gap_s} s"
    )

    print(
        f"Status JSONL: "
        f"{status_jsonl_path}"
    )

    print("=" * 60, flush=True)

    # ============================================================
    # PROCESSES
    # ============================================================

    slam_proc = None
    metrics_proc = None
    status_log_proc = None
    bag_proc = None

    try:
        # --------------------------------------------------------
        # SLAM
        # --------------------------------------------------------

        slam_proc = start_group(
            [
                sys.executable,
                str(run_slam),
                "--config",
                str(config_path),
                "--use-sim-time",
            ],
            cwd=root,
        )

        # --------------------------------------------------------
        # AGGREGATED METRICS RECORDER
        # --------------------------------------------------------

        metrics_proc = start_group(
            [
                sys.executable,
                str(recorder),
                "--algorithm",
                algorithm,
                "--variant",
                variant,
                "--config-file",
                str(config_path),
                "--bag-name",
                bag_path.name,
                "--run-id",
                safe_run_id,
                "--output-dir",
                str(
                    (
                        root
                        / args.output_dir
                    ).resolve()
                ),
                "--odom-topic",
                odom_topic,
                "--status-topic",
                status_topic,
                "--jump-translation-m",
                str(
                    args.jump_translation_m
                ),
                "--jump-yaw-deg",
                str(
                    args.jump_yaw_deg
                ),
                "--alignment-max-gap-s",
                str(
                    args.alignment_max_gap_s
                ),
            ],
            cwd=root,
        )

        # --------------------------------------------------------
        # FULL FRAME-BY-FRAME STATUS LOGGER
        # --------------------------------------------------------

        status_log_proc = start_group(
            [
                sys.executable,
                "-u",
                "-c",
                STATUS_LOGGER_CODE,
                status_topic,
                str(status_jsonl_path),
            ],
            cwd=root,
        )

        # Give all ROS nodes time to create their subscriptions before
        # starting rosbag playback.
        time.sleep(
            args.startup_delay
        )

        if (
            status_log_proc.poll()
            is not None
        ):
            raise RuntimeError(
                "Status JSONL logger exited before bag playback."
            )

        # --------------------------------------------------------
        # ROSBAG
        # --------------------------------------------------------

        # By default replay only the topics needed for the SLAM benchmark
        # plus the RGB camera used as visual ground truth/reference in
        # Foxglove. Use --all-topics to replay the complete bag.
        bag_cmd = [
            "ros2",
            "bag",
            "play",
            str(bag_path),
            "--clock",
            "--rate",
            str(args.rate),
        ]

        if not args.all_topics:
            bag_cmd.extend(
                [
                    "--topics",
                    *args.topics,
                ]
            )

        bag_proc = start_group(
            bag_cmd,
            cwd=root,
        )

        bag_code = (
            bag_proc.wait()
        )

        # --------------------------------------------------------
        # DRAIN CALLBACKS
        # --------------------------------------------------------

        # Allow queued LiDAR / SLAM callbacks to drain after bag playback.
        time.sleep(
            args.drain_delay
        )

        # Stop recorders only after the callbacks have drained, otherwise
        # the last SLAM statuses could be lost.
        terminate_group(
            metrics_proc,
            signal.SIGINT,
            timeout=8.0,
        )
        metrics_proc = None

        terminate_group(
            status_log_proc,
            signal.SIGINT,
            timeout=8.0,
        )
        status_log_proc = None

        if bag_code != 0:
            raise SystemExit(
                bag_code
            )

        # ========================================================
        # UPDATE METADATA
        # ========================================================

        metadata["finished_at"] = (
            datetime.now()
            .astimezone()
            .isoformat()
        )

        metadata["bag_exit_code"] = int(
            bag_code
        )

        try:
            with status_jsonl_path.open(
                "r",
                encoding="utf-8",
            ) as status_file:
                status_message_count = sum(
                    1
                    for line in status_file
                    if line.strip()
                )

        except FileNotFoundError:
            status_message_count = 0

        metadata[
            "status_message_count"
        ] = int(
            status_message_count
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ========================================================
        # SUMMARY
        # ========================================================

        print()
        print(
            "Bag finished. Metrics saved in:"
        )

        print(
            (
                root
                / args.output_dir
                / "tfm_slam_benchmark.csv"
            ).resolve()
        )

        if global_opt_enabled:
            print(
                "Expected summary rows: RAW + OPTIMIZED "
                "(if optimized path was published)."
            )
        else:
            print(
                "Expected summary rows: RAW only."
            )

        print(
            f"Detailed run: "
            f"{run_dir}"
        )

        print(
            f"Frame-by-frame status: "
            f"{status_jsonl_path}"
        )

        print(
            f"Status messages recorded: "
            f"{status_message_count}"
        )

        if args.keep_alive:
            print()
            print("=" * 60)
            print("Bag playback finished.")
            print("SLAM nodes remain alive for Foxglove inspection.")
            print("Press Ctrl+C when finished.")
            print("=" * 60, flush=True)

            while True:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print(
            "Interrupted by user.",
            flush=True,
        )

    finally:
        terminate_group(
            bag_proc,
            signal.SIGINT,
        )

        terminate_group(
            metrics_proc,
            signal.SIGINT,
            timeout=8.0,
        )

        terminate_group(
            status_log_proc,
            signal.SIGINT,
            timeout=8.0,
        )

        terminate_group(
            slam_proc,
            signal.SIGINT,
            timeout=8.0,
        )


if __name__ == "__main__":
    main()