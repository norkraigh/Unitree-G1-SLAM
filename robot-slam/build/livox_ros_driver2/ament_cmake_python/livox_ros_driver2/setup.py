from setuptools import find_packages
from setuptools import setup

setup(
    name='livox_ros_driver2',
    version='0.0.1',
    packages=find_packages(
        include=('livox_ros_driver2', 'livox_ros_driver2.*')),
)
