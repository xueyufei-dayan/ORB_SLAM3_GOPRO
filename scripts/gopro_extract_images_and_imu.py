#!/usr/bin/env python3
"""Extract decoded GoPro frames and GPMF IMU telemetry on a Unix-time clock.

The frames are written without resizing, undistortion, or colour conversion.
Their filenames are Unix timestamps in nanoseconds, for example
``1712345678123456789.png``.  The generated JSON has the same fields and IMU
axis convention as :mod:`gopro_extract_imu`, except all timestamp fields are
Unix timestamps in nanoseconds instead of timestamps relative to the video.

Example:
    python scripts/gopro_extract_images_and_imu.py /mnt/d/data/gopro/op1/GX010111.MP4

This creates ``/mnt/d/data/gopro/op1/GX010111/images`` and
``/mnt/d/data/gopro/op1/GX010111/imu.json``.  GoPro's MP4 ``creation_time`` is
used as the Unix-time origin.  Supply ``--start-time`` when that tag is absent
or incorrect.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# This script lives next to gopro_extract_imu.py, so this works both when it is
# executed directly and when it is imported by another script.
from gopro_extract_imu import (apply_axis_alignment, extract_telemetry,
                               synchronize_to_accl, trim_both_ends)


NS_PER_S = 1_000_000_000


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
    return int(round(parsed.timestamp() * NS_PER_S))


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


def write_images(video, timestamps_ns, image_dir, extension, jpeg_quality):
    """Decode each displayed frame and save it under its Unix-ns timestamp."""
    image_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")

    options = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if extension == "jpg" else []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index >= len(timestamps_ns):
                raise RuntimeError("OpenCV decoded more frames than ffprobe reported")
            output = image_dir / f"{int(timestamps_ns[index])}.{extension}"
            if not cv2.imwrite(str(output), image, options):
                raise RuntimeError(f"failed to write image: {output}")
            index += 1
    finally:
        capture.release()

    if index != len(timestamps_ns):
        raise RuntimeError(f"OpenCV decoded {index} frames but ffprobe reported "
                           f"{len(timestamps_ns)}; no complete result was produced")
    return index


def make_telemetry(video, origin_ns, frame_timestamps_ns, skip_seconds, axis_align):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp4", type=Path, help="GoPro MP4 video")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: <video-stem>/ beside the video)")
    parser.add_argument("--imu-output", type=Path, default=None,
                        help="IMU JSON path (default: <output-dir>/imu.json)")
    parser.add_argument("--start-time", type=float, default=None,
                        help="Recording start Unix time in seconds; overrides MP4 creation_time")
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
    if not 0 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in 0..100")

    output_dir = args.output_dir or args.mp4.with_suffix("")
    imu_path = args.imu_output or output_dir / "imu.json"
    try:
        probe = run_ffprobe(args.mp4)
        relative_frame_ns = video_frame_times_ns(probe)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        sys.exit(f"[ERROR] Cannot read video timestamps with ffprobe: {error}")

    if args.start_time is not None:
        origin_ns = int(round(args.start_time * NS_PER_S))
    else:
        tags = probe.get("format", {}).get("tags", {})
        origin_ns = parse_creation_time(tags.get("creation_time"))
        if origin_ns is None:
            sys.exit("[ERROR] MP4 has no valid creation_time; pass --start-time UNIX_SECONDS")

    frame_timestamps_ns = origin_ns + relative_frame_ns
    print(f"[INFO] Unix-time origin: {origin_ns} ns")
    print(f"[INFO] Writing {len(frame_timestamps_ns)} images to {output_dir / 'images'}")
    try:
        written = write_images(args.mp4, frame_timestamps_ns, output_dir / "images",
                               args.format, args.jpeg_quality)
        telemetry = make_telemetry(args.mp4, origin_ns, frame_timestamps_ns,
                                   args.skip_seconds, not args.no_axis_align)
    except RuntimeError as error:
        sys.exit(f"[ERROR] {error}")

    imu_path.parent.mkdir(parents=True, exist_ok=True)
    with imu_path.open("w", encoding="utf-8") as handle:
        json.dump(telemetry, handle, indent=2)
    print(f"[INFO] Wrote {written} images and IMU JSON -> {imu_path}")


if __name__ == "__main__":
    main()
