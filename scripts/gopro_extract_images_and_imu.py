#!/usr/bin/env python3
"""Extract undistorted GoPro frames and synchronized GPMF IMU telemetry.

The frames are fisheye-undistorted using an ORB-SLAM3 camera-settings file and
written without resizing or colour conversion.  Their filenames are Unix
timestamps in nanoseconds, for example
``1712345678123456789.png``.  The generated JSON has the same fields and IMU
axis convention as OpenICC.  All timestamp fields are Unix timestamps in
nanoseconds.  Every image filename also occurs in ``img_timestamps_ns`` and
the IMU ``timestamps_ns`` values use the same Unix clock.

Example:
    python scripts/gopro_extract_images_and_imu.py video.MP4 \
        --output-dir dataset \
        --settings calibration.yaml

Images are written to ``<output-dir>/images`` and telemetry is written to
``<output-dir>/imu.json``.  The output directory and calibration file are
supplied explicitly; there are no built-in data or calibration paths.  GoPro's
MP4 ``creation_time`` is used as the Unix-time origin.  Supply ``--start-time``
when that tag is absent or incorrect.
"""

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path

import cv2
import numpy as np

NS_PER_S = 1_000_000_000


def to_seconds(timestamps, label):
    """Auto-detect a telemetry timestamp unit and convert it to seconds."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return timestamps
    span = timestamps[-1] - timestamps[0]
    if span < 1e4:
        return timestamps
    if span < 1e6:
        print(f"[INFO] {label} timestamps are milliseconds; converting to seconds")
        return timestamps / 1e3
    if span < 1e9:
        print(f"[INFO] {label} timestamps are microseconds; converting to seconds")
        return timestamps / 1e6
    print(f"[INFO] {label} timestamps are nanoseconds; converting to seconds")
    return timestamps / NS_PER_S


def synchronize_to_accl(accl_ts, accl, gyro_ts, gyro):
    """Synchronize GYRO samples onto the ACCL timestamp sequence."""
    if len(accl_ts) == len(gyro_ts) and np.allclose(accl_ts, gyro_ts):
        return accl_ts, accl, gyro
    print("[INFO] ACCL/GYRO timestamps differ; interpolating GYRO onto ACCL")
    gyro_sync = np.column_stack(
        [np.interp(accl_ts, gyro_ts, gyro[:, axis]) for axis in range(3)])
    return accl_ts, accl, gyro_sync


def apply_axis_alignment(gyro, accl):
    """Convert raw GoPro axes to the OpenICC calibration frame."""
    return gyro[:, [1, 2, 0]], accl[:, [1, 2, 0]]


def trim_both_ends(timestamps, accl, gyro, skip_seconds):
    """Remove skip_seconds from both ends of synchronized IMU data."""
    if skip_seconds <= 0 or len(timestamps) < 3:
        return timestamps, accl, gyro
    interval = timestamps[1] - timestamps[0]
    count = int(round(skip_seconds / interval))
    if count <= 0:
        return timestamps, accl, gyro
    if 2 * count >= len(timestamps):
        raise RuntimeError(f"--skip-seconds {skip_seconds} removes all IMU samples")
    selected = slice(count, len(timestamps) - count)
    print(f"[INFO] Trimmed {skip_seconds}s ({count} IMU samples at each end)")
    return timestamps[selected], accl[selected], gyro[selected]


def extract_telemetry(mp4_path):
    """Extract image timestamps and available GPMF sensor streams."""
    try:
        from py_gpmf_parser.gopro_telemetry_extractor import (  # pylint: disable=import-outside-toplevel
            GoProTelemetryExtractor,
        )
    except ImportError as error:
        raise RuntimeError(
            "py-gpmf-parser is required to extract GoPro telemetry") from error

    extractor = GoProTelemetryExtractor(str(mp4_path))
    extractor.open_source()
    try:
        image_timestamps = np.asarray(
            extractor.get_image_timestamps_s(), dtype=np.float64)
        streams = {}
        for sensor in ("ACCL", "GYRO", "GRAV", "CORI", "GPS5", "GPSP"):
            print(f"[INFO] Extracting {sensor} stream")
            try:
                data, timestamps = extractor.extract_data(sensor)
            except Exception as error:  # Optional streams vary between cameras.
                print(f"[WARN] {sensor} extraction failed: {error}")
                continue
            data = np.asarray(data, dtype=np.float64)
            if len(data):
                streams[sensor] = (data, to_seconds(timestamps, sensor))
    finally:
        extractor.close_source()
    return image_timestamps, streams


def run_ffprobe(video):
    """Return ffprobe JSON with recording metadata and displayed-frame PTSes."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format_tags=creation_time:frame=best_effort_timestamp_time",
        "-show_frames", "-of", "json", str(video),
    ]
    result = subprocess.run(command, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)


def parse_creation_time(value):
    """Parse ISO-8601 ffprobe creation_time into integral Unix nanoseconds."""
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta = parsed.astimezone(dt.timezone.utc) - epoch
    return ((delta.days * 86400 + delta.seconds) * NS_PER_S
            + delta.microseconds * 1000)


def unix_seconds_to_ns(value):
    """Parse decimal Unix seconds without losing nanoseconds to float rounding."""
    try:
        seconds = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a Unix timestamp in seconds") from error
    if not seconds.is_finite():
        raise argparse.ArgumentTypeError("must be a finite Unix timestamp")
    return int((seconds * NS_PER_S).to_integral_value(rounding=ROUND_HALF_EVEN))


def video_frame_times_ns(probe):
    """Get per-frame relative timestamps from ffprobe, preserving variable FPS."""
    times = []
    for frame in probe.get("frames", []):
        value = frame.get("best_effort_timestamp_time")
        if value is not None:
            times.append(int(round(float(value) * NS_PER_S)))
    if not times:
        raise RuntimeError("ffprobe returned no video-frame timestamps")
    first = times[0]
    relative = np.asarray(times, dtype=np.int64) - first
    if np.any(np.diff(relative) <= 0):
        raise RuntimeError("video-frame timestamps are not strictly increasing")
    return relative


def read_fisheye_calibration(settings):
    """Read camera intrinsics and camera-to-IMU extrinsics from settings."""
    storage = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"cannot open camera settings: {settings}")
    try:
        camera_type = storage.getNode("Camera.type").string()
        if camera_type != "KannalaBrandt8":
            raise RuntimeError(
                f"Camera.type must be KannalaBrandt8, got {camera_type!r}")

        def value(name):
            node = storage.getNode(name)
            if node.empty() or (not node.isReal() and not node.isInt()):
                raise RuntimeError(f"missing numeric setting {name} in {settings}")
            return node.real()

        # New ORB-SLAM3 settings use Camera1.*, while older files use Camera.*.
        prefix = "Camera1" if not storage.getNode("Camera1.fx").empty() else "Camera"
        matrix = np.array([
            [value(f"{prefix}.fx"), 0.0, value(f"{prefix}.cx")],
            [0.0, value(f"{prefix}.fy"), value(f"{prefix}.cy")],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        distortion = np.asarray(
            [value(f"{prefix}.k{index}") for index in range(1, 5)],
            dtype=np.float64).reshape(4, 1)
        calibration_size = (int(value("Camera.width")),
                            int(value("Camera.height")))
        extrinsics_node = storage.getNode("IMU.T_b_c1")
        extrinsics = None if extrinsics_node.empty() else extrinsics_node.mat()
    finally:
        storage.release()

    if calibration_size[0] <= 0 or calibration_size[1] <= 0:
        raise RuntimeError(f"invalid calibration image size: {calibration_size}")
    if extrinsics is None or extrinsics.shape != (4, 4):
        raise RuntimeError(f"missing 4x4 IMU.T_b_c1 extrinsics in {settings}")
    return matrix, distortion, calibration_size, extrinsics


def make_undistort_maps(settings, image_size):
    """Undistort into a pinhole image with the input image's intrinsics."""
    camera_matrix, distortion, calibration_size, _ = read_fisheye_calibration(settings)
    scale_x = image_size[0] / calibration_size[0]
    scale_y = image_size[1] / calibration_size[1]
    camera_matrix[0, :] *= scale_x
    camera_matrix[1, :] *= scale_y
    camera_matrix[2, 2] = 1.0

    return cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, distortion, np.eye(3), camera_matrix,
        image_size, cv2.CV_32FC1)


def write_images(video, timestamps_ns, image_dir, extension, jpeg_quality,
                 settings):
    """Undistort displayed frames and save them under their Unix-ns timestamps."""
    image_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")

    options = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if extension == "jpg" else []
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"invalid decoded video size: {width}x{height}")
    map_x, map_y = make_undistort_maps(settings, (width, height))
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index >= len(timestamps_ns):
                raise RuntimeError("OpenCV decoded more frames than ffprobe reported")
            output = image_dir / f"{int(timestamps_ns[index])}.{extension}"
            undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            if not cv2.imwrite(str(output), undistorted, options):
                raise RuntimeError(f"failed to write image: {output}")
            index += 1
    finally:
        capture.release()

    if index != len(timestamps_ns):
        raise RuntimeError(f"OpenCV decoded {index} frames but ffprobe reported "
                           f"{len(timestamps_ns)}; no complete result was produced")
    return index


def make_telemetry(video, origin_ns, frame_timestamps_ns, skip_seconds, axis_align,
                   extrinsics=None):
    """Create gopro_extract_imu-compatible JSON, using Unix nanoseconds."""
    _, streams = extract_telemetry(str(video))
    if "ACCL" not in streams or "GYRO" not in streams:
        raise RuntimeError("ACCL or GYRO stream not found in the video")

    accl, accl_ts = streams["ACCL"]
    gyro, gyro_ts = streams["GYRO"]
    ts, accl, gyro = synchronize_to_accl(accl_ts, accl, gyro_ts, gyro)
    if axis_align:
        gyro, accl = apply_axis_alignment(gyro, accl)
    ts, accl, gyro = trim_both_ends(ts, accl, gyro, skip_seconds)

    out = {
        "accelerometer": accl.tolist(),
        "gyroscope": gyro.tolist(),
        "timestamps_ns": (origin_ns + np.rint(ts * NS_PER_S).astype(np.int64)).tolist(),
        # Use the same PTSes used for image names, rather than assuming a
        # constant frame rate or a telemetry-library-specific frame clock.
        "img_timestamps_ns": frame_timestamps_ns.astype(np.int64).tolist(),
        "camera_fps": (NS_PER_S / np.mean(np.diff(frame_timestamps_ns))
                       if len(frame_timestamps_ns) > 1 else 0.0),
    }
    if "GRAV" in streams:
        out["gravity"] = streams["GRAV"][0].tolist()
    if "CORI" in streams:
        out["camera_orientation"] = streams["CORI"][0][:, [1, 3, 2, 0]].tolist()
    if "GPS5" in streams:
        gps, gps_ts = streams["GPS5"]
        out["gps_llh"] = gps[:, :3].tolist()
        out["gps_timestamps_ns"] = (origin_ns + np.rint(gps_ts * NS_PER_S).astype(np.int64)).tolist()
        if "GPSP" in streams:
            out["gps_precision"] = streams["GPSP"][0][:, 0].tolist()
    return out


def write_imu_csv(path, telemetry):
    """Write synchronized absolute-time IMU samples as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp_ns", "gx", "gy", "gz", "ax", "ay", "az"))
        for timestamp, gyro, accl in zip(
                telemetry["timestamps_ns"], telemetry["gyroscope"],
                telemetry["accelerometer"]):
            writer.writerow((int(timestamp), *gyro, *accl))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp4", type=Path, help="GoPro MP4 video")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="Output directory (<dir>/images and <dir>/imu.json)")
    parser.add_argument("--imu-csv", type=Path,
                        help="Also write synchronized IMU samples to this CSV")
    parser.add_argument("--start-time", type=unix_seconds_to_ns, default=None,
                        help="Recording start Unix time in seconds; overrides MP4 creation_time")
    parser.add_argument("--settings", type=Path,
                        help="External YAML containing Camera1.* intrinsics and IMU.T_b_c1 "
                             "extrinsics (required unless --imu-only)")
    parser.add_argument("--imu-only", action="store_true",
                        help="Extract telemetry without decoding or writing images")
    parser.add_argument("--skip-seconds", type=float, default=0.0,
                        help="Trim this duration from both IMU ends")
    parser.add_argument("--no-axis-align", action="store_true",
                        help="Keep raw GoPro axes instead of the [y,z,x] permutation")
    parser.add_argument("--format", choices=("png", "jpg"), default="png",
                        help="Image format; png is lossless (default)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality, used only with --format jpg (0..100)")
    args = parser.parse_args()

    if not args.mp4.is_file():
        parser.error(f"video not found: {args.mp4}")
    if not args.imu_only and args.settings is None:
        parser.error("--settings is required unless --imu-only is used")
    if args.settings is not None and not args.settings.is_file():
        parser.error(f"camera settings not found: {args.settings}")
    if args.skip_seconds < 0.0:
        parser.error("--skip-seconds must be non-negative")
    if not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in 0..100")

    image_dir = args.output_dir / "images"
    imu_path = args.output_dir / "imu.json"
    try:
        probe = run_ffprobe(args.mp4)
        relative_frame_ns = video_frame_times_ns(probe)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        sys.exit(f"[ERROR] Cannot read video timestamps with ffprobe: {error}")

    if args.start_time is not None:
        origin_ns = args.start_time
    else:
        tags = probe.get("format", {}).get("tags", {})
        origin_ns = parse_creation_time(tags.get("creation_time"))
        if origin_ns is None:
            sys.exit("[ERROR] MP4 has no valid creation_time; pass --start-time UNIX_SECONDS")

    frame_timestamps_ns = origin_ns + relative_frame_ns
    print(f"[INFO] Unix-time origin: {origin_ns} ns")
    try:
        extrinsics = (read_fisheye_calibration(args.settings)[3]
                      if args.settings is not None else None)
        telemetry = make_telemetry(args.mp4, origin_ns, frame_timestamps_ns,
                                   args.skip_seconds, not args.no_axis_align,
                                   extrinsics)
        written = 0
        if not args.imu_only:
            print(f"[INFO] Writing {len(frame_timestamps_ns)} undistorted images "
                  f"to {image_dir}")
            written = write_images(
                args.mp4, frame_timestamps_ns, image_dir, args.format,
                args.jpeg_quality, args.settings)
    except (RuntimeError, cv2.error) as error:
        sys.exit(f"[ERROR] {error}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with imu_path.open("w", encoding="utf-8") as handle:
        json.dump(telemetry, handle, indent=2)
    if args.imu_csv is not None:
        write_imu_csv(args.imu_csv, telemetry)
        print(f"[INFO] Wrote IMU CSV -> {args.imu_csv}")
    if args.imu_only:
        print(f"[INFO] Wrote IMU JSON -> {imu_path}")
    else:
        print(f"[INFO] Wrote {written} images and IMU JSON -> {imu_path}")


if __name__ == "__main__":
    main()
