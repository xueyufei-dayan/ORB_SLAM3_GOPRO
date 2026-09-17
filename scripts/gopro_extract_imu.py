#!/usr/bin/env python3
"""
Extract IMU telemetry from a GoPro MP4 and write the JSON format consumed by
Examples/Monocular-Inertial/mono_inertial_gopro_vi.cc.

Single-script replacement for the OpenICC two-step pipeline
"GoProTelemetryExtractor.extract_data_to_json() + TelemetryConverter.convert_pygpmf_telemetry()".

Uses GoProTelemetryExtractor from the py-gpmf-parser package.
Conda environment: vigs-slam-5090

Usage:
    conda activate vigs-slam-5090
    python scripts/gopro_extract_imu.py GX01xxxx.MP4              # -> GX01xxxx.json
    python scripts/gopro_extract_imu.py GX01xxxx.MP4 -o imu.json --skip-seconds 2
    python scripts/gopro_extract_imu.py GX01xxxx.MP4 --no-axis-align --csv imu.csv

Output JSON keys (same as TelemetryConverter output):
    accelerometer, gyroscope   : [x, y, z] per sample, m/s^2 and rad/s
    timestamps_ns              : ns since recording start (ACCL timebase)
    img_timestamps_ns          : video frame timestamps, ns
    camera_fps                 : video frame rate
    gravity                    : optional GRAV stream (camera frame)
    camera_orientation         : optional CORI stream, quaternion [x, y, z, w]
    gps_llh / gps_precision / gps_timestamps_ns : optional GPS streams

Axis convention:
    Raw GoPro IMU frame: x right, y down, z forward (out of the lens).
    By default axes are permuted to [new_x, new_y, new_z] = [old_y, old_z, old_x]
    to match the OpenICC calibration frame of Examples/Monocular-Inertial/gopro9_wide_setting.yaml
    (same permutation as OpenImuCameraCalibrator's TelemetryConverter).
    Use --no-axis-align to keep raw GoPro axes.
"""

import argparse
import json
import os
import sys

import numpy as np


def to_seconds(ts, label):
    """Auto-detect timestamp unit (s/ms/us/ns) and convert to seconds."""
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 2:
        return ts
    rng = ts[-1] - ts[0]
    if rng < 1e4:
        return ts
    if rng < 1e6:
        print(f"[INFO] {label} timestamps detected as milliseconds -> converting to seconds")
        return ts / 1e3
    if rng < 1e9:
        print(f"[INFO] {label} timestamps detected as microseconds -> converting to seconds")
        return ts / 1e6
    print(f"[INFO] {label} timestamps detected as nanoseconds -> converting to seconds")
    return ts / 1e9


def synchronize_to_accl(accl_ts, accl, gyro_ts, gyro):
    """Sync GYRO onto ACCL timestamps (interpolate if rates/timestamps differ)."""
    if len(accl_ts) == len(gyro_ts) and np.allclose(accl_ts, gyro_ts):
        return accl_ts, accl, gyro
    print("[INFO] ACCL and GYRO timestamps differ: interpolating GYRO onto ACCL timebase")
    gyro_sync = np.column_stack(
        [np.interp(accl_ts, gyro_ts, gyro[:, i]) for i in range(3)])
    return accl_ts, accl, gyro_sync


def apply_axis_alignment(gyro, accl):
    """Permute raw GoPro axes -> calibration frame: [new] = [old_y, old_z, old_x]."""
    gyro_aligned = gyro[:, [1, 2, 0]]
    accl_aligned = accl[:, [1, 2, 0]]
    return gyro_aligned, accl_aligned


def trim_both_ends(ts, accl, gyro, skip_seconds):
    """Remove skip_seconds from the start AND end of the stream (OpenICC semantics)."""
    if skip_seconds <= 0 or len(ts) < 3:
        return ts, accl, gyro
    dt = ts[1] - ts[0]
    n = int(round(skip_seconds / dt))
    if n <= 0:
        return ts, accl, gyro
    if 2 * n >= len(ts):
        print(f"[ERROR] --skip-seconds {skip_seconds} removes the whole stream")
        sys.exit(1)
    sl = slice(n, len(ts) - n)
    print(f"[INFO] Trimmed {skip_seconds}s from start and end ({n} samples each side)")
    return ts[sl], accl[sl], gyro[sl]


def extract_telemetry(mp4_path):
    """Extract all telemetry streams from the MP4.

    Returns:
        img_ts  : (F,) video frame timestamps in seconds
        streams : {sensor: (data (N,d), timestamps_s (N,))} for ACCL, GYRO
                  and optionally GRAV, CORI, GPS5, GPSP
    """
    try:
        from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
    except ImportError:
        print("[ERROR] py-gpmf-parser is not installed in this environment.")
        print("        Activate the conda environment first:")
        print("        conda activate vigs-slam-5090")
        sys.exit(1)

    if not os.path.exists(mp4_path):
        print(f"[ERROR] File not found: {mp4_path}")
        sys.exit(1)

    print(f"[INFO] Opening MP4: {mp4_path}")
    extractor = GoProTelemetryExtractor(mp4_path)
    extractor.open_source()
    try:
        img_ts = np.asarray(extractor.get_image_timestamps_s(), dtype=np.float64)

        streams = {}
        for sensor in ("ACCL", "GYRO", "GRAV", "CORI", "GPS5", "GPSP"):
            print(f"[INFO] Extracting {sensor} stream...")
            try:
                data, ts = extractor.extract_data(sensor)
            except Exception as e:  # sensor stream may not exist for this file
                print(f"[WARN] {sensor} extraction failed: {e}")
                continue
            data = np.asarray(data, dtype=np.float64)
            ts = to_seconds(ts, sensor)
            if len(data) == 0:
                print(f"[WARN] {sensor} stream is empty, skipping")
                continue
            streams[sensor] = (data, ts)
    finally:
        extractor.close_source()

    return img_ts, streams


def main():
    parser = argparse.ArgumentParser(
        description="Extract GoPro IMU telemetry from an MP4 -> JSON for "
                    "mono_inertial_gopro_vi.cc (replaces py_gpmf_parser + TelemetryConverter)")
    parser.add_argument('mp4', type=str, help='Path to the GoPro MP4 video file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON path (default: <mp4>.json next to the video)')
    parser.add_argument('--skip-seconds', type=float, default=0.0,
                        help='Seconds to cut from the start AND end of the IMU stream '
                             '(OpenICC TelemetryConverter semantics)')
    parser.add_argument('--no-axis-align', action='store_true',
                        help='Keep raw GoPro axes (skip [y,z,x] permutation)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Also write synchronized IMU CSV: timestamp,gx,gy,gz,ax,ay,az')
    args = parser.parse_args()

    # 1. Extract all streams
    img_ts, streams = extract_telemetry(args.mp4)

    if "ACCL" not in streams or "GYRO" not in streams:
        print("[ERROR] ACCL or GYRO stream not found in the video")
        sys.exit(1)
    accl_data, accl_ts = streams["ACCL"]
    gyro_data, gyro_ts = streams["GYRO"]

    # 2. Synchronize GYRO onto ACCL timebase
    ts, accl, gyro = synchronize_to_accl(accl_ts, accl_data, gyro_ts, gyro_data)

    # 3. Axis alignment (default: OpenICC calibration frame)
    if args.no_axis_align:
        print("[INFO] Kept raw GoPro axes (no permutation)")
    else:
        gyro, accl = apply_axis_alignment(gyro, accl)
        print("[INFO] Applied axis permutation [new_x,new_y,new_z] = [old_y, old_z, old_x]")

    # 4. Trim start/end
    ts, accl, gyro = trim_both_ends(ts, accl, gyro, args.skip_seconds)

    n = len(ts)
    duration = ts[-1] - ts[0] if n > 1 else 0.0
    print(f"[INFO] ACCL: {n} samples, {ts[0]:.3f}s - {ts[-1]:.3f}s "
          f"({duration:.1f}s, ~{n / duration:.0f} Hz)" if n > 1 else
          f"[INFO] ACCL: {n} samples")

    # 5. Assemble output dict (same structure as TelemetryConverter output)
    out = {}
    out["accelerometer"] = accl.tolist()
    out["gyroscope"] = gyro.tolist()
    out["timestamps_ns"] = (ts * 1e9).tolist()

    img_ts_ns = img_ts * 1e9
    out["img_timestamps_ns"] = img_ts_ns.tolist()
    out["camera_fps"] = 1.0 / np.mean(np.diff(img_ts)) if len(img_ts) > 1 else 0.0

    if "GRAV" in streams:
        out["gravity"] = streams["GRAV"][0].tolist()
    if "CORI" in streams:
        cori = streams["CORI"][0]
        # GoPro CORI order w,x,z,y -> output quaternion [x,y,z,w]
        # https://github.com/gopro/gpmf-parser/issues/100#issuecomment-656154136
        out["camera_orientation"] = cori[:, [1, 3, 2, 0]].tolist()
    if "GPS5" in streams:
        gps_data, gps_ts = streams["GPS5"]
        out["gps_llh"] = gps_data[:, :3].tolist()  # lat, long, alt
        out["gps_timestamps_ns"] = (gps_ts * 1e9).tolist()
        if "GPSP" in streams:
            out["gps_precision"] = streams["GPSP"][0][:, 0].tolist()

    # 6. Write JSON
    out_path = args.output if args.output else os.path.splitext(args.mp4)[0] + ".json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Wrote telemetry JSON -> {out_path}")

    # 7. Optional CSV
    if args.csv:
        imu_csv = np.zeros((n, 7), dtype=np.float64)
        imu_csv[:, 0] = ts
        imu_csv[:, 1:4] = gyro
        imu_csv[:, 4:7] = accl
        np.savetxt(args.csv, imu_csv, delimiter=',',
                   header='timestamp,gx,gy,gz,ax,ay,az', comments='',
                   fmt=['%.9f', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e'])
        print(f"[INFO] Wrote synchronized IMU CSV -> {args.csv}")


if __name__ == '__main__':
    main()
