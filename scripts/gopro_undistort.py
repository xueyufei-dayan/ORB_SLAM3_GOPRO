#!/usr/bin/env python3
"""Undistort a GoPro MP4 using an ORB-SLAM3 camera-settings YAML file.

Example:
    python3 scripts/gopro_undistort.py \
        /mnt/d/data/gopro/op1/GX010111.MP4 \
        Examples/Monocular-Inertial/gopro9_wide_setting.yaml

This writes ``/mnt/d/data/gopro/op1/GX010111_undistort.MP4`` by default.
The image dimensions and frame rate are retained.  Audio is copied from the
source MP4 when ffmpeg is available.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def require_opencv():
    try:
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise SystemExit(
            "OpenCV Python bindings are required. Activate the SLAM conda "
            "environment or install python3-opencv.") from error
    return cv2, np


def read_calibration(cv2, settings):
    storage = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot open settings file: {settings}")
    try:
        camera_type = storage.getNode("Camera.type").string()
        if camera_type != "KannalaBrandt8":
            raise ValueError(
                f"Only Camera.type KannalaBrandt8 is supported, got {camera_type!r}")

        def value(name):
            node = storage.getNode(name)
            if node.empty():
                raise ValueError(f"Missing {name} in {settings}")
            return node.real()

        width = int(value("Camera.width"))
        height = int(value("Camera.height"))
        matrix = ((value("Camera.fx"), value("Camera.fy"),
                   value("Camera.cx"), value("Camera.cy")), width, height)
        distortion = [value(f"Camera.k{index}") for index in range(1, 5)]
        return matrix, distortion
    finally:
        storage.release()


def mux_audio(ffmpeg, silent_video, source_video, destination):
    command = [ffmpeg, "-y", "-v", "error", "-i", str(silent_video),
               "-i", str(source_video), "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart",
               str(destination)]
    subprocess.run(command, check=True)


def undistort(video, settings, destination, balance):
    cv2, np = require_opencv()
    ((fx, fy, cx, cy), calibration_width, calibration_height), distortion = read_calibration(cv2, settings)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Invalid video properties: {width}x{height} at {fps} FPS")

        scale_x = width / calibration_width
        scale_y = height / calibration_height
        camera_matrix = np.array([[fx * scale_x, 0.0, cx * scale_x],
                                  [0.0, fy * scale_y, cy * scale_y],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(4, 1)
        rectified_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, distortion, (width, height), np.eye(3), balance=balance)
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, distortion, np.eye(3), rectified_matrix,
            (width, height), cv2.CV_32FC1)

        with tempfile.TemporaryDirectory(prefix="gopro_undistort_") as temp_dir:
            silent_video = Path(temp_dir) / "video.mp4"
            writer = cv2.VideoWriter(
                str(silent_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create temporary MP4 video with OpenCV")
            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    writer.write(cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT))
                    index += 1
                    if index % 100 == 0:
                        print(f"[INFO] Undistorted {index}/{frame_count} frames", flush=True)
            finally:
                writer.release()

            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                mux_audio(ffmpeg, silent_video, video, destination)
            else:
                shutil.move(silent_video, destination)
                print("[WARN] ffmpeg not found; output contains video only", flush=True)
    finally:
        capture.release()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Input GoPro MP4")
    parser.add_argument("settings", type=Path, help="ORB-SLAM3 KannalaBrandt8 YAML settings")
    parser.add_argument("-o", "--output", type=Path, default=None, required=True,
                        help="Output MP4 (default: <input_stem>_undistort<suffix>)")
    parser.add_argument("--balance", type=float, default=0.0,
                        help="OpenCV fisheye balance in [0, 1]; 0 crops black borders (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.video.is_file():
        sys.exit(f"Video not found: {args.video}")
    if not args.settings.is_file():
        sys.exit(f"Settings not found: {args.settings}")
    if not 0.0 <= args.balance <= 1.0:
        sys.exit("--balance must be in [0, 1]")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Undistorting {args.video} -> {output}", flush=True)
    undistort(args.video, args.settings, output, args.balance)
    print(f"[DONE] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
