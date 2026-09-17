# ORB-SLAM3 Docker 使用说明

## 1. 构建镜像

在工程根目录执行：

```bash
docker build --build-arg BUILD_JOBS=4 -t orb-slam3:latest .
```

`BUILD_JOBS` 是编译并发数。内存不足时改为 `2`，内存充足时可以增大。

也可以使用 Compose：

```bash
mkdir -p data output
docker compose build
```

## 2. 启动交互式容器

ORB-SLAM3 使用 Pangolin 显示图形窗口。在 Linux/X11 主机上先授权本地容器访问 X Server：

```bash
xhost +local:docker
```

然后启动容器：

```bash
docker run --rm -it \
  --name orb-slam3 \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /宿主机/数据集目录:/data:ro \
  -v "$(pwd)/output:/output" \
  orb-slam3:latest
```

退出后建议收回授权：

```bash
xhost -local:docker
```

使用 Compose 启动时，可通过环境变量指定宿主机目录：

```bash
ORB_SLAM3_DATASET_DIR=/宿主机/数据集目录 \
ORB_SLAM3_OUTPUT_DIR="$(pwd)/output" \
docker compose run --rm orb-slam3
```

Wayland 桌面通常也提供 XWayland，因此可使用上述方式。远程或无桌面环境需要 X11 转发、VNC，或者在代码配置中关闭 Viewer。

## 3. 运行示例

进入容器后，当前目录是 `/opt/ORB_SLAM3`。以 EuRoC 单目示例为例：

```bash
./Examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/EuRoC.yaml \
  /data/MH01 \
  Examples/Monocular/EuRoC_TimeStamps/MH01.txt
```

运行工程中的 GoPro 示例：

```bash
./Examples/Monocular-Inertial/gopro_slam --help
```

镜像已经包含 `ffmpeg`、OpenCV Python 模块和 `py-gpmf-parser`。

## 4. 直接执行命令

也可以不进入 Shell，直接把程序参数写在镜像名后面：

```bash
docker run --rm -it \
  --ipc=host \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /宿主机/数据集目录:/data:ro \
  orb-slam3:latest \
  ./Examples/Monocular/mono_euroc \
  Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml \
  /data/MH01 Examples/Monocular/EuRoC_TimeStamps/MH01.txt
```

## 5. 摄像头和 GPU（可选）

USB 摄像头通常需要增加设备映射：

```bash
--device=/dev/video0:/dev/video0
```

若使用 Intel RealSense，通常还需要映射 USB 总线并安装对应依赖：

```bash
--device=/dev/bus/usb:/dev/bus/usb
```

当前 ORB-SLAM3 构建使用 CPU。若后续加入 CUDA，再安装 NVIDIA Container Toolkit，并在 `docker run` 中增加 `--gpus all`。

## 6. 常用维护命令

```bash
# 查看镜像
docker images orb-slam3

# 删除镜像
docker image rm orb-slam3:latest

# 不使用缓存重新构建
docker build --no-cache --build-arg BUILD_JOBS=4 -t orb-slam3:latest .
```
