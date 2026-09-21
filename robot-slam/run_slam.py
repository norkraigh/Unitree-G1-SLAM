import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROCESSES = []


# ============================================================
# PROCESS MANAGEMENT
# ============================================================

def terminate_children():
    """
    Stops all processes started by this launcher.

    Processes are started in their own process groups so that ROS 2 launch
    children are also terminated when run_slam.py exits.
    """

    for proc in reversed(PROCESSES):
        if proc.poll() is None:
            try:
                os.killpg(
                    os.getpgid(proc.pid),
                    signal.SIGTERM,
                )
            except (ProcessLookupError, PermissionError):
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass

    deadline = time.time() + 3.0

    for proc in reversed(PROCESSES):
        if proc.poll() is None:
            timeout = max(
                0.0,
                deadline - time.time(),
            )

            try:
                proc.wait(timeout=timeout)

            except subprocess.TimeoutExpired:
                try:
                    os.killpg(
                        os.getpgid(proc.pid),
                        signal.SIGKILL,
                    )
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass


def stop_children(*_args):
    """
    SIGINT / SIGTERM handler.
    """

    print(
        "\nStopping SLAM processes...",
        flush=True,
    )

    terminate_children()

    raise SystemExit(0)


def start(cmd, cwd=None):
    """
    Starts a child process and stores it in the global process list.
    """

    print(
        "+",
        " ".join(str(x) for x in cmd),
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        start_new_session=True,
    )

    PROCESSES.append(proc)

    return proc


def monitor_processes():
    """
    Keeps run_slam.py alive while all children are running.

    If any SLAM component exits unexpectedly, all remaining processes are
    stopped so that the system does not remain in a partially running state.
    """

    while True:

        for proc in list(PROCESSES):

            code = proc.poll()

            if code is not None:

                print(
                    f"Process PID={proc.pid} exited "
                    f"with code {code}.",
                    flush=True,
                )

                terminate_children()

                raise SystemExit(code)

        time.sleep(0.2)


# ============================================================
# HELPERS
# ============================================================

def ros_package_prefix(package_name):
    """
    Returns the ROS installation prefix for a package.
    """

    check = subprocess.run(
        [
            "ros2",
            "pkg",
            "prefix",
            package_name,
        ],
        capture_output=True,
        text=True,
    )

    if check.returncode != 0:
        return None

    prefix = check.stdout.strip()

    if not prefix:
        return None

    return Path(prefix)


def bool_string(value):
    """
    Converts Python bool-like values to ROS parameter strings.
    """

    return "true" if bool(value) else "false"


def resolve_project_path(root, value):
    """
    Resolve a path from the project root unless an absolute path is supplied.
    """

    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


# ============================================================
# CAMERA COMPRESSION FOR FOXGLOVE
# ============================================================

def launch_camera_compression(root):
    """
    Republishes the raw RGB camera topic as a compressed image topic.

    This keeps the original /camera/color/image_raw topic available inside
    ROS 2 while providing /camera/color/image_raw/compressed for Foxglove,
    reducing the bandwidth sent through the bridge.
    """

    print(
        "[CAMERA] Starting compressed image transport",
        flush=True,
    )

    start(
        [
            "ros2",
            "run",
            "image_transport",
            "republish",
            "raw",
            "compressed",

            "--ros-args",

            "-r",
            "in:=/camera/color/image_raw",

            "-r",
            "out/compressed:=/camera/color/image_raw/compressed",
        ],
        cwd=root,
    )


# ============================================================
# COMMON LIVOX MOUNT TF
# ============================================================

def launch_livox_mount_tf(
    root,
    config_path,
    use_sim_time,
):
    """
    Starts the common static TF publisher for the Livox mounting correction.

    The physical mounting transform is defined once in
    config/livox_slam_config.json and is shared by all front-ends.
    """

    node = (
        root
        / "scripts"
        / "livox_mount_tf_publisher.py"
    )

    if not node.exists():
        raise SystemExit(
            f"Livox mount TF publisher not found: {node}"
        )

    print(
        "[SENSOR] Starting common Livox mount TF publisher",
        flush=True,
    )

    start(
        [
            sys.executable,
            str(node),
            "--ros-args",
            "-p",
            f"config_file:={config_path}",
            "-p",
            f"use_sim_time:={use_sim_time}",
        ],
        cwd=root,
    )


# ============================================================
# ICP PROPIO
# ============================================================

def launch_icp(
    root,
    config_path,
    use_sim_time,
):
    """
    Starts the custom ICP frontend.
    """

    node = (
        root
        / "src"
        / "g1_lidar_odometry"
        / "g1_lidar_odometry"
        / "livox_slam_pose_node.py"
    )

    if not node.exists():
        raise SystemExit(
            f"ICP node not found: {node}"
        )

    print(
        "[SLAM] Starting custom ICP",
        flush=True,
    )

    start(
        [
            sys.executable,
            str(node),

            "--ros-args",

            "-p",
            f"config_file:={config_path}",

            "-p",
            f"use_sim_time:={use_sim_time}",
        ],
        cwd=root,
    )


# ============================================================
# KISS-ICP
# ============================================================

def launch_kiss_icp(
    root,
    config_path,
    config,
    algorithm_options,
    use_sim_time,
):
    """
    Starts KISS-ICP and its /g1/slam/* output adapter.

    The KISS core configuration is read from a project-owned YAML file rather
    than from the package's installed default. This makes the experiment
    reproducible and allows MID-360-specific tuning.
    """

    kiss_cfg = (
        algorithm_options
        .get("kiss_icp", {})
    )

    kiss_prefix = ros_package_prefix(
        "kiss_icp"
    )

    if kiss_prefix is None:

        raise SystemExit(
            "KISS-ICP is not built or the workspace "
            "has not been sourced.\n"
            "Run:\n"
            "  source install/setup.bash"
        )

    kiss_cfg_name = kiss_cfg.get(
        "config_file",
        "config/kiss_icp_mid360.yaml",
    )

    kiss_params_file = resolve_project_path(
        root,
        kiss_cfg_name,
    )

    if not kiss_params_file.exists():

        raise SystemExit(
            "KISS-ICP project config file not found: "
            f"{kiss_params_file}"
        )

    topics = config.get(
        "topics",
        {},
    )

    frames = config.get(
        "frames",
        {},
    )

    debug_cfg = config.get(
        "debug",
        {},
    )

    reset_cfg = config.get(
        "reset",
        {},
    )

    topic = kiss_cfg.get(
        "topic",
        topics.get(
            "lidar",
            "/livox/lidar",
        ),
    )

    livox_cfg = (
        config
        .get("sensors", {})
        .get("livox", {})
    )

    mount_correction_enabled = bool(
        livox_cfg.get(
            "apply_mount_correction",
            False,
        )
    )

    default_base_frame = (
        frames.get(
            "corrected_lidar_frame",
            "livox_corrected_frame",
        )
        if mount_correction_enabled
        else frames.get(
            "robot_frame",
            "livox_frame",
        )
    )

    base_frame = kiss_cfg.get(
        "base_frame",
        default_base_frame,
    )

    raw_lidar_frame = frames.get(
        "robot_frame",
        "livox_frame",
    )

    lidar_odom_frame = kiss_cfg.get(
        "lidar_odom_frame",
        frames.get(
            "fixed_frame",
            "map",
        ),
    )

    publish_odom_tf = bool_string(
        kiss_cfg.get(
            "publish_odom_tf",
            True,
        )
    )

    invert_odom_tf = bool_string(
        kiss_cfg.get(
            "invert_odom_tf",
            False,
        )
    )

    publish_debug_clouds = bool_string(
        kiss_cfg.get(
            "publish_debug_clouds",
            True,
        )
    )

    print(
        "[SLAM] Starting KISS-ICP",
        flush=True,
    )

    print(
        f"[SLAM] Input topic: {topic}",
        flush=True,
    )

    print(
        f"[SLAM] Base frame: {base_frame}",
        flush=True,
    )

    print(
        f"[SLAM] Odometry frame: {lidar_odom_frame}",
        flush=True,
    )

    print(
        f"[SLAM] KISS config: {kiss_params_file}",
        flush=True,
    )

    adapter = (
        root
        / "scripts"
        / "kiss_icp_output_adapter.py"
    )

    if not adapter.exists():

        raise SystemExit(
            "KISS-ICP output adapter not found: "
            f"{adapter}"
        )

    start(
        [
            sys.executable,
            str(adapter),

            "--ros-args",

            "-p",
            f"fixed_frame:={lidar_odom_frame}",

            "-p",
            f"config_file:={config_path}",

            "-p",
            f"base_frame:={base_frame}",

            "-p",
            f"raw_lidar_frame:={raw_lidar_frame}",

            "-p",
            "odom_input_topic:=/kiss/odometry",

            "-p",
            "local_map_input_topic:=/kiss/local_map",

            "-p",
            "frame_input_topic:=/kiss/frame",

            "-p",
            (
                "max_path_length:="
                f"{int(debug_cfg.get('max_path_length', 5000))}"
            ),

            "-p",
            (
                "publish_status:="
                f"{bool_string(debug_cfg.get('publish_status', True))}"
            ),

            "-p",
            (
                "clear_path_on_clock_jump:="
                f"{bool_string(reset_cfg.get('clear_path', True))}"
            ),

            "-p",
            (
                "clock_jump_threshold_sec:="
                f"{float(reset_cfg.get('clock_jump_threshold_sec', 1.0))}"
            ),

            "-p",
            f"use_sim_time:={use_sim_time}",
        ],
        cwd=root,
    )

    start(
        [
            "ros2",
            "run",
            "kiss_icp",
            "kiss_icp_node",

            "--ros-args",

            "-r",
            f"pointcloud_topic:={topic}",

            "--params-file",
            str(kiss_params_file),

            "-p",
            f"base_frame:={base_frame}",

            "-p",
            f"lidar_odom_frame:={lidar_odom_frame}",

            "-p",
            f"publish_odom_tf:={publish_odom_tf}",

            "-p",
            f"invert_odom_tf:={invert_odom_tf}",

            "-p",
            f"publish_debug_clouds:={publish_debug_clouds}",

            "-p",
            f"use_sim_time:={use_sim_time}",
        ],
        cwd=root,
    )


# ============================================================
# FAST-LIO2
# ============================================================

def launch_fast_lio2(
    root,
    config_path,
    config,
    algorithm_options,
    use_sim_time,
):
    """
    Starts FAST-LIO2 and its /g1/slam/* output adapter.

    The adapter receives both:

      fast_lio_config
          Native FAST-LIO2 LiDAR/IMU configuration.

      config_file
          Main project configuration containing the common Livox mounting
          correction.
    """

    fast_cfg = (
        algorithm_options
        .get("fast_lio2", {})
    )

    # --------------------------------------------------------
    # FAST-LIO2 configuration file
    # --------------------------------------------------------

    fast_cfg_name = fast_cfg.get(
        "config_file",
        "config/fast_lio2_mid360.yaml",
    )

    fast_cfg_path = (
        (root / fast_cfg_name).resolve()
        if not Path(fast_cfg_name).is_absolute()
        else Path(fast_cfg_name).resolve()
    )

    if not fast_cfg_path.exists():

        raise SystemExit(
            "FAST-LIO2 config not found: "
            f"{fast_cfg_path}"
        )

    # --------------------------------------------------------
    # Check installation
    # --------------------------------------------------------

    if ros_package_prefix(
        "fast_lio"
    ) is None:

        raise SystemExit(
            "FAST-LIO2 is not built in this workspace. "
            "Run ./install_fast_lio2.sh first, "
            "then source install/setup.bash."
        )

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------

    frames = config.get(
        "frames",
        {},
    )

    fixed_frame = frames.get(
        "fixed_frame",
        "map",
    )

    internal_fixed_frame = fast_cfg.get(
        "internal_fixed_frame",
        "camera_init",
    )

    body_frame = fast_cfg.get(
        "body_frame",
        "body",
    )

    lidar_frame = fast_cfg.get(
        "lidar_frame",
        frames.get(
            "robot_frame",
            "livox_frame",
        ),
    )

    corrected_lidar_frame = frames.get(
        "corrected_lidar_frame",
        "livox_corrected_frame",
    )

    print(
        "[SLAM] Starting FAST-LIO2",
        flush=True,
    )

    print(
        f"[SLAM] Fixed frame: {fixed_frame}",
        flush=True,
    )

    print(
        f"[SLAM] FAST-LIO2 internal frame: "
        f"{internal_fixed_frame}",
        flush=True,
    )

    print(
        f"[SLAM] FAST-LIO2 native config: "
        f"{fast_cfg_path}",
        flush=True,
    )

    print(
        f"[SLAM] Project config: "
        f"{config_path}",
        flush=True,
    )

    # --------------------------------------------------------
    # FAST-LIO2 output adapter
    # --------------------------------------------------------

    adapter = (
        root
        / "scripts"
        / "fast_lio_output_adapter.py"
    )

    if not adapter.exists():

        raise SystemExit(
            "FAST-LIO2 output adapter not found: "
            f"{adapter}"
        )

    start(
        [
            sys.executable,
            str(adapter),

            "--ros-args",

            "-p",
            f"fixed_frame:={fixed_frame}",

            "-p",
            (
                "internal_fixed_frame:="
                f"{internal_fixed_frame}"
            ),

            "-p",
            f"body_frame:={body_frame}",

            "-p",
            f"lidar_frame:={lidar_frame}",

            "-p",
            (
                "corrected_lidar_frame:="
                f"{corrected_lidar_frame}"
            ),

            # Native FAST-LIO2 configuration.
            "-p",
            f"fast_lio_config:={fast_cfg_path}",

            # Main project configuration.
            # The adapter reads sensors.livox.rotation_deg and
            # sensors.livox.translation_m from this file.
            "-p",
            f"config_file:={config_path}",

            "-p",
            f"use_sim_time:={use_sim_time}",
        ],
        cwd=root,
    )

    # --------------------------------------------------------
    # FAST-LIO2
    # --------------------------------------------------------

    start(
        [
            "ros2",
            "launch",
            "fast_lio",
            "mapping.launch.py",

            f"config_path:={fast_cfg_path.parent}",

            f"config_file:={fast_cfg_path.name}",

            "rviz:=false",

            f"use_sim_time:={use_sim_time}",
        ],
        cwd=root,
    )


# ============================================================
# COMMON GLOBAL BACKEND
# ============================================================

def launch_global_backend(
    root,
    config_path,
    config,
    use_sim_time,
):
    """
    Starts the common loop-closure + pose-graph backend.

    All three frontends feed the same normalized interface:

        /g1/slam/odom
        /g1/slam/aligned_cloud

    The optimized results are published separately under:

        /g1/slam/optimized/*
    """

    global_cfg = config.get(
        "global_optimization",
        {},
    )

    enabled = bool(
        global_cfg.get(
            "enabled",
            False,
        )
    )

    if not enabled:

        print(
            "[BACKEND] Global optimization: disabled",
            flush=True,
        )

        return

    backend_algorithm = str(
        global_cfg.get(
            "algorithm",
            "pose_graph",
        )
    ).lower()

    if backend_algorithm != "pose_graph":

        raise SystemExit(
            "Unsupported global optimization algorithm: "
            f"{backend_algorithm}\n"
            "Available backend algorithms: pose_graph"
        )

    backend = (
        root
        / "scripts"
        / "global_pose_graph_backend.py"
    )

    if not backend.exists():

        raise SystemExit(
            "Global pose graph backend not found: "
            f"{backend}"
        )

    backend_nice = max(0, int(global_cfg.get("process_nice", 0)))

    print(
        "[BACKEND] Starting loop closure + "
        "pose graph optimization"
        + (f" (nice={backend_nice})" if backend_nice > 0 else ""),
        flush=True,
    )

    backend_cmd = [
        sys.executable,
        str(backend),
        "--ros-args",
        "-p",
        f"config_file:={config_path}",
        "-p",
        f"use_sim_time:={use_sim_time}",
    ]

    # Lower backend CPU priority so loop closure cannot starve the real-time
    # frontend. Positive niceness is permitted for an unprivileged user.
    if backend_nice > 0:
        backend_cmd = ["nice", "-n", str(backend_nice), *backend_cmd]

    start(
        backend_cmd,
        cwd=root,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/livox_slam_config.json",
    )

    parser.add_argument(
        "--use-sim-time",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    root = Path(
        __file__
    ).resolve().parent

    config_argument = Path(
        args.config
    )

    config_path = (
        (root / config_argument).resolve()
        if not config_argument.is_absolute()
        else config_argument.resolve()
    )

    if not config_path.exists():

        raise SystemExit(
            f"Config file not found: {config_path}"
        )

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    slam_cfg = config.get(
        "slam",
        {},
    )

    algorithm = str(
        slam_cfg.get(
            "algorithm",
            "icp",
        )
    ).lower()

    algorithm_options = slam_cfg.get(
        "algorithm_options",
        {},
    )

    use_sim_time = (
        "true"
        if args.use_sim_time
        else "false"
    )

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    signal.signal(
        signal.SIGINT,
        stop_children,
    )

    signal.signal(
        signal.SIGTERM,
        stop_children,
    )

    # --------------------------------------------------------
    # General information
    # --------------------------------------------------------

    print(
        "============================================================",
        flush=True,
    )

    print(
        "TFM G1 SLAM",
        flush=True,
    )

    print(
        f"Algorithm: {algorithm}",
        flush=True,
    )

    print(
        f"Config: {config_path}",
        flush=True,
    )

    print(
        f"use_sim_time: {use_sim_time}",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    # --------------------------------------------------------
    # Common sensor geometry
    # --------------------------------------------------------

    launch_livox_mount_tf(
        root=root,
        config_path=config_path,
        use_sim_time=use_sim_time,
    )

    # Publish a compressed copy of the RGB camera stream for Foxglove.
    # The raw topic remains available internally and is still the source
    # recorded/replayed by rosbag2.
    launch_camera_compression(
        root=root,
    )

    # Give DDS/TF a brief wall-clock interval to discover the static
    # transform before the first LiDAR frame reaches KISS-ICP.
    time.sleep(0.5)

    # --------------------------------------------------------
    # Select frontend
    # --------------------------------------------------------

    if algorithm == "icp":

        launch_icp(
            root=root,
            config_path=config_path,
            use_sim_time=use_sim_time,
        )

    elif algorithm == "kiss_icp":

        launch_kiss_icp(
            root=root,
            config_path=config_path,
            config=config,
            algorithm_options=algorithm_options,
            use_sim_time=use_sim_time,
        )

    elif algorithm == "fast_lio2":

        launch_fast_lio2(
            root=root,
            config_path=config_path,
            config=config,
            algorithm_options=algorithm_options,
            use_sim_time=use_sim_time,
        )

    else:

        raise SystemExit(
            f"Unsupported slam.algorithm: {algorithm}\n"
            "Available algorithms: "
            "icp, kiss_icp, fast_lio2"
        )

    # --------------------------------------------------------
    # Common backend
    # --------------------------------------------------------

    launch_global_backend(
        root=root,
        config_path=config_path,
        config=config,
        use_sim_time=use_sim_time,
    )

    # --------------------------------------------------------
    # Keep everything alive
    # --------------------------------------------------------

    monitor_processes()


if __name__ == "__main__":
    main()