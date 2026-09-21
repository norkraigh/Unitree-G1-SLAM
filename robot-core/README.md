# robot-core

Workspace del robot para adquisición de sensores.

Actualmente solo es necesario cuando se quieren generar datos nuevos. Para reproducir bags y desarrollar SLAM en el PC, no se usa.

## Compilar

```bash
cd ~/robot-core
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Arrancar sensores

```bash
cd ~/robot-core
./start_g1_full_sensor_stack.sh
```

Publicaciones relevantes para SLAM:
- `/livox/lidar`
- `/livox/imu`

La RealSense se mantiene en el stack de adquisición aunque el SLAM ICP actual no la consume.
