#!/usr/bin/env python3
"""Patch the ROS2 FAST-LIO branch to consume Livox MID-360 PointCloud2
per-point timestamps directly.

Livox ROS Driver 2 PointXYZRTLT contains:
  x y z intensity tag line timestamp
where timestamp is the absolute point timestamp. FAST-LIO expects each
point's relative time in milliseconds in PointType.curvature.
"""
from pathlib import Path
import re
import sys


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("Usage: patch_fast_lio2_mid360.py <FAST_LIO_source_dir>")

    root = Path(sys.argv[1]).resolve()
    header = root / "src" / "preprocess.h"
    source = root / "src" / "preprocess.cpp"

    if not header.exists() or not source.exists():
        fail(f"FAST-LIO2 source not found under {root}")

    h = header.read_text(encoding="utf-8")
    cpp = source.read_text(encoding="utf-8")

    if "LivoxPointXyzrtlt" in h and "point.timestamp - base_time_ns" in cpp:
        print("FAST-LIO2 MID360 timestamp patch already applied.")
        return

    header_pattern = re.compile(
        r"namespace livox_ros\s*\{\s*typedef struct \{.*?\}\s*LivoxPointXyzrtl;\s*\}\s*"
        r"POINT_CLOUD_REGISTER_POINT_STRUCT\(livox_ros::LivoxPointXyzrtl,.*?\n\)\s*",
        re.S,
    )

    header_replacement = r'''namespace livox_ros
{
typedef struct {
  float x;
  float y;
  float z;
  float intensity;
  uint8_t tag;
  uint8_t line;
  double timestamp;
} LivoxPointXyzrtlt;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_ros::LivoxPointXyzrtlt,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint8_t, tag, tag)
    (uint8_t, line, line)
    (double, timestamp, timestamp)
)
'''

    h2, n = header_pattern.subn(header_replacement, h, count=1)
    if n != 1:
        fail("Could not locate Livox PointCloud2 point definition in preprocess.h. Upstream FAST-LIO2 may have changed.")

    # The patched handler uses std::isfinite explicitly. Do not rely on
    # transitive includes from ROS/PCL.
    if "#include <cmath>" not in cpp:
        cpp = cpp.replace('#include "preprocess.h"', '#include "preprocess.h"\n#include <cmath>', 1)

    func_pattern = re.compile(
        r"void Preprocess::mid360_handler\(const sensor_msgs::msg::PointCloud2::UniquePtr &msg\)\s*\{.*?\n\}\s*"
        r"(?=void Preprocess::default_handler)",
        re.S,
    )

    func_replacement = r'''void Preprocess::mid360_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<livox_ros::LivoxPointXyzrtlt> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    const int plsize = static_cast<int>(pl_orig.points.size());
    if (plsize == 0)
        return;

    pl_surf.reserve(plsize);

    // Livox ROS Driver 2 sets PointCloud2.header.stamp to the packet base time
    // and the per-point `timestamp` field to the absolute point time (ns).
    // FAST-LIO stores the relative point time in `curvature`, in milliseconds.
    const double base_time_ns = static_cast<double>(
        rclcpp::Time(msg->header.stamp).nanoseconds());

    for (int i = 0; i < plsize; ++i)
    {
        const auto &point = pl_orig.points[i];
        const int layer = static_cast<int>(point.line);

        if (layer < 0 || layer >= N_SCANS)
            continue;

        if (i % point_filter_num != 0)
            continue;

        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = point.x;
        added_pt.y = point.y;
        added_pt.z = point.z;
        added_pt.intensity = point.intensity;

        double offset_ns = point.timestamp - base_time_ns;
        if (!std::isfinite(offset_ns) || offset_ns < 0.0)
            offset_ns = 0.0;

        added_pt.curvature = offset_ns * 1e-6;  // ns -> ms

        if (added_pt.x * added_pt.x +
            added_pt.y * added_pt.y +
            added_pt.z * added_pt.z > blind * blind)
        {
            pl_surf.push_back(std::move(added_pt));
        }
    }
}

'''

    cpp2, n = func_pattern.subn(func_replacement, cpp, count=1)
    if n != 1:
        fail("Could not locate mid360_handler in preprocess.cpp. Upstream FAST-LIO2 may have changed.")

    header.write_text(h2, encoding="utf-8")
    source.write_text(cpp2, encoding="utf-8")
    print("Applied FAST-LIO2 MID360 PointCloud2 timestamp patch.")


if __name__ == "__main__":
    main()
