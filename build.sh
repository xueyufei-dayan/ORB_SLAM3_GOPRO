#!/usr/bin/env bash

set -e

BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${BUILD_JOBS}"

cd ../../Pangolin
echo "Configuring and building Thirdparty/Pangolin ..."
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF
make -j"${BUILD_JOBS}"

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPangolin_DIR="$(pwd)/../Thirdparty/Pangolin/build"
make -j"${BUILD_JOBS}"
