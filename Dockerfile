FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Build and runtime dependencies for ORB-SLAM3, Pangolin and the GoPro helper.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        libavcodec-dev \
        libavdevice-dev \
        libavformat-dev \
        libavutil-dev \
        libboost-dev \
        libboost-serialization-dev \
        libegl1-mesa-dev \
        libeigen3-dev \
        libepoxy-dev \
        libgl1-mesa-dev \
        libglew-dev \
        libjpeg-dev \
        libopencv-dev \
        libpng-dev \
        libssl-dev \
        libswscale-dev \
        libwayland-dev \
        libxkbcommon-dev \
        ninja-build \
        python3 \
        python3-numpy \
        python3-opencv \
        python3-pip \
        wayland-protocols \
    && python3 -m pip install --no-cache-dir py-gpmf-parser \
    && rm -rf /var/lib/apt/lists/*

ARG BUILD_JOBS=4

ENV TZ=Asia/Shanghai \
    LD_LIBRARY_PATH=/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/opt/ORB_SLAM3/Thirdparty/Pangolin/build \
    PYTHONUNBUFFERED=1

WORKDIR /opt/ORB_SLAM3

COPY . .

RUN BUILD_JOBS="${BUILD_JOBS}" bash ./build.sh

CMD ["/bin/bash"]
