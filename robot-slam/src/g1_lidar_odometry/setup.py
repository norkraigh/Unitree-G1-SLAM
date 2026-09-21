from setuptools import setup

package_name = "g1_lidar_odometry"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ugiviag1",
    maintainer_email="tacaniki@hotmail.com",
    description="Unitree G1 Livox ICP SLAM with IMU prior and optional pose-graph optimization",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "livox_slam_pose_node = g1_lidar_odometry.livox_slam_pose_node:main",
        ],
    },
)
