# TFM Unitree G1 - estructura activa

## Qué workspace usar

### `robot-core`
Se usa **en el robot** únicamente cuando se quieren adquirir datos nuevos.

Responsabilidad actual:
- arrancar el Livox MID-360 (`/livox/lidar`, `/livox/imu`);
- arrancar la Intel RealSense;
- publicar sensores brutos por ROS 2.

No contiene el algoritmo SLAM.

### `robot-slam`
Se usa **en el PC** para procesar los sensores/bags y ejecutar el SLAM.

Implementación activa:
- ICP scan-to-map;
- corrección de montaje Livox;
- ayuda de yaw de la IMU;
- keyframes y mapa voxelizado;
- Pose Graph + loop closure + optimización global;
- `/g1/slam/path` y `/g1/slam/optimized_path`.

El nodo vigente es:

```text
robot-slam/src/g1_lidar_odometry/g1_lidar_odometry/livox_slam_pose_node.py
```

## Flujo offline actual

Para trabajar con el bag base no hace falta ejecutar `robot-core`.

### Terminal 1 - SLAM

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam
./run_icp_slam.sh
```

### Terminal 2 - Foxglove

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project
source g1-slam/activate_g1_slam_pc_humble.sh
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

### Terminal 3 - bag base

```bash
cd ~/Documents/TFM-Miquel_Reynes/Follower_Project
source /opt/ros/humble/setup.bash
ros2 bag play bags/bag-slam-largo-2026-07-14 --clock
```

Foxglove:
- Fixed frame: `map`
- `/g1/slam/local_map`
- `/g1/slam/aligned_cloud`
- `/g1/slam/path`
- `/g1/slam/optimized_path`

## Flujo para grabar un bag nuevo

En el robot se usa `robot-core/start_g1_full_sensor_stack.sh` para publicar los sensores. El SLAM no se ejecuta en este workspace.

## FAST-LIO2

No se incluye todavía en esta copia activa. El ZIP original contenía configuraciones parciales de FAST-LIO2 pero no una instalación/nodo integrado completo; se han eliminado para evitar que parezcan operativas. FAST-LIO2 debe añadirse después como alternativa separada al front-end ICP, manteniendo esta versión como baseline reproducible.
