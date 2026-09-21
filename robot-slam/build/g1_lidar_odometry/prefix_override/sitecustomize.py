import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ugiviag1/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam/install/g1_lidar_odometry'
