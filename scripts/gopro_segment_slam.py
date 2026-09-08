#!/usr/bin/env python3
"""Run GoPro inertial SLAM in overlapping frame windows and chain atlas maps.

Each window contains 15,000 frames by default; consecutive windows advance by
14,000 frames, so they overlap by 1,000 frames.  The atlas written by a window
is passed to the next window with ``--load_map``.

The output layout follows ``.../slam_v3``::

    output/
      0_15000/mapping_camera_trajectory.csv
      0_15000/map_atlas.osa
      0_15000/slam_stdout.txt
      0_15000/slam_stderr.txt
      14000_29000/...
      mapping_camera_trajectory.csv     # merged, non-overlapping trajectory
      mapping_camera_trajectory_tum.txt # merged TUM trajectory
      slices.json                       # [[start_frame, end_frame], ...]

An optional IMU JSON can be supplied; otherwise the script invokes
gopro_extract_imu.py to extract telemetry next to the GoPro MP4 first.  It uses
ffmpeg to make frame-exact temporary MP4s and creates a matching, time-rebased
IMU JSON for every window.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLAM = REPO_ROOT / "Examples/Monocular-Inertial/gopro_slam"
DEFAULT_SETTINGS = REPO_ROOT / "Examples/Monocular-Inertial/gopro9_wide_setting.yaml"
DEFAULT_VOCABULARY = REPO_ROOT / "Vocabulary/ORBvoc.txt"


def run(command, *, cwd, stdout_path=None, stderr_path=None):
    """Run a command, optionally directing its output to two log files."""
    print("[RUN]", " ".join(str(part) for part in command), flush=True)
    if stdout_path is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=True)


def video_info(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate,r_frame_rate",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    frame_count = int(stream["nb_read_frames"])
    fps_text = stream.get("avg_frame_rate", "0/0")
    if fps_text == "0/0":
        fps_text = stream["r_frame_rate"]
    numerator, denominator = (int(value) for value in fps_text.split("/"))
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"Cannot determine FPS from {video}")
    return frame_count, numerator / denominator


def load_and_slice_imu(imu_path, start_seconds, end_seconds, destination):
    """Write the IMU samples needed by one video window, with time zero at it."""
    with imu_path.open() as source:
        data = json.load(source)
    required = ("accelerometer", "gyroscope", "timestamps_ns")
    if not all(key in data for key in required):
        raise ValueError("IMU JSON must contain accelerometer, gyroscope and timestamps_ns; "
                         "generate it with scripts/gopro_extract_imu.py first")

    timestamps = data["timestamps_ns"]
    accel = data["accelerometer"]
    gyro = data["gyroscope"]
    if not timestamps or len(timestamps) != len(accel) or len(timestamps) != len(gyro):
        raise ValueError("IMU timestamps, accelerometer and gyroscope arrays must be non-empty and equal length")

    origin_ns = float(timestamps[0])
    selected = []
    # gopro_slam's frame loop looks at the next IMU timestamp without an
    # end-bound check, so retain a small tail beyond the final video frame.
    # This also gives the last frame its closing IMU integration interval.
    tail_seconds = 1.0
    # Include one measurement immediately before the window for integration.
    for index, timestamp in enumerate(timestamps):
        relative_seconds = (float(timestamp) - origin_ns) * 1e-9
        if relative_seconds <= end_seconds + tail_seconds:
            if relative_seconds >= start_seconds:
                selected.append(index)
        else:
            break
    if not selected:
        raise ValueError(f"No IMU samples found in video interval {start_seconds:.6f}..{end_seconds:.6f}s")
    first = max(0, selected[0] - 1)
    selected = list(range(first, selected[-1] + 1))
    # gopro_slam zeroes timestamps from the first item.  Keeping that first
    # sample close to the video boundary prevents a large initial IMU gap.
    first_ns = float(timestamps[selected[0]])
    out = {
        "accelerometer": [accel[index] for index in selected],
        "gyroscope": [gyro[index] for index in selected],
        "timestamps_ns": [float(timestamps[index]) - first_ns for index in selected],
    }
    # A recording may end just before the final video frame.  Add a harmless
    # hold sample in that case so gopro_slam never advances beyond its IMU
    # arrays (its current loop has no bounds check on that index).
    final_frame_seconds = end_seconds - start_seconds
    if out["timestamps_ns"][-1] * 1e-9 <= final_frame_seconds:
        out["accelerometer"].append(out["accelerometer"][-1])
        out["gyroscope"].append(out["gyroscope"][-1])
        out["timestamps_ns"].append(out["timestamps_ns"][-1] + tail_seconds * 1e9)
    destination.write_text(json.dumps(out))


def make_segment_video(video, start, end, destination):
    # select works on decoded frame indexes, unlike -ss/-t which is keyframe based.
    expression = f"select='between(n\\,{start}\\,{end - 1})',setpts=N/FRAME_RATE/TB"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", expression,
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(destination)],
        cwd=REPO_ROOT)


def filter_stable_rows(rows, stable_frames, max_speed):
    """Drop poses after tracking loss/jumps until motion is stable again."""
    accepted = []
    pending = []
    previous = None
    stable = False

    for row in rows:
        if row["is_lost"].lower() == "true":
            stable = False
            pending.clear()
            previous = None
            continue

        continuous = True
        if previous is not None:
            dt = float(row["timestamp"]) - float(previous["timestamp"])
            distance = math.sqrt(sum(
                (float(row[key]) - float(previous[key])) ** 2
                for key in ("x", "y", "z")))
            continuous = dt > 0.0 and distance / dt <= max_speed

        if not continuous:
            stable = False
            pending = [row]
        elif stable:
            accepted.append(row)
        else:
            pending.append(row)
            if len(pending) >= stable_frames:
                accepted.extend(pending)
                pending.clear()
                stable = True
        previous = row

    return accepted


def merge_trajectories(output_dir, windows, fps, stable_frames, max_stable_speed):
    destination = output_dir / "mapping_camera_trajectory.csv"
    tum_destination = output_dir / "mapping_camera_trajectory_tum.txt"
    fieldnames = None
    merged_rows = []

    for window_index, (start, end) in enumerate(windows):
        path = output_dir / f"{start}_{end}" / "mapping_camera_trajectory.csv"
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                local_frame = int(row["frame_idx"])
                if window_index and local_frame < windows[window_index - 1][1] - start:
                    continue  # discard the overlap already emitted by the prior window
                row["frame_idx"] = str(start + local_frame)
                row["timestamp"] = f"{start / fps + float(row['timestamp']):.9f}"
                merged_rows.append(row)

    #stable_rows = filter_stable_rows(merged_rows, stable_frames, max_stable_speed)
    with destination.open("w", newline="") as target, tum_destination.open("w") as tum_target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        #for row in stable_rows:
        for row in merged_rows:
            writer.writerow(row)
            tum_target.write(
                f"{row['timestamp']} {row['x']} {row['y']} {row['z']} "
                f"{row['q_x']} {row['q_y']} {row['q_z']} {row['q_w']}\n")

    #removed = len(merged_rows) - len(stable_rows)
    #print(f"[INFO] Stability filter removed {removed} poses; kept {len(stable_rows)}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="GoPro MP4 input")
    parser.add_argument("imu_json", type=Path, nargs="?",
                        help="Optional IMU JSON from gopro_extract_imu.py (default: extract from video)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: <video_stem>_slam_v3)")
    parser.add_argument("--slam", type=Path, default=DEFAULT_SLAM, help="gopro_slam executable")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="ORB-SLAM settings YAML")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY, help="ORB vocabulary")
    parser.add_argument("--window-frames", type=int, default=15000)
    parser.add_argument("--overlap-frames", type=int, default=1000)
    parser.add_argument("--stable-frames", type=int, default=30,
                        help="Consecutive continuous poses required after tracking loss/jump (default: 30)")
    parser.add_argument("--max-stable-speed", type=float, default=20.0,
                        help="Maximum translation speed considered continuous, m/s (default: 20)")
    parser.add_argument("--mask-img", type=Path)
    parser.add_argument("--enable-gui", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip windows that already have trajectory and atlas files")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_frames <= 0 or args.overlap_frames < 0 or args.overlap_frames >= args.window_frames:
        sys.exit("--window-frames must be positive and --overlap-frames must be in [0, window-frames)")
    if args.stable_frames <= 0 or args.max_stable_speed <= 0:
        sys.exit("--stable-frames and --max-stable-speed must be positive")
    for path, label in ((args.video, "video"), (args.slam, "gopro_slam"), (args.settings, "settings"),
                        (args.vocabulary, "vocabulary")):
        if not path.is_file():
            sys.exit(f"{label} not found: {path}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg and ffprobe must be installed and available on PATH")

    output_dir = args.output_dir or args.video.with_suffix("").with_name(args.video.stem + "_slam_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / ".segments"
    temporary_dir.mkdir(exist_ok=True)
    if args.imu_json is None:
        args.imu_json = args.video.with_suffix(".json")
        if args.imu_json.is_file():
            print(f"[INFO] Reusing IMU JSON next to video: {args.imu_json}", flush=True)
        else:
            print(f"[INFO] No IMU JSON supplied; extracting GoPro telemetry to {args.imu_json}", flush=True)
            run([sys.executable, str(REPO_ROOT / "scripts/gopro_extract_imu.py"),
                 str(args.video.resolve()), "--output", str(args.imu_json.resolve())], cwd=REPO_ROOT)
    elif not args.imu_json.is_file():
        sys.exit(f"IMU JSON not found: {args.imu_json}")
    frame_count, fps = video_info(args.video)
    step = args.window_frames - args.overlap_frames
    windows = [(start, min(start + args.window_frames, frame_count))
               for start in range(0, frame_count, step)]
    print(f"[INFO] {frame_count} frames at {fps:.9g} FPS; {len(windows)} windows", flush=True)
    (output_dir / "slices.json").write_text(json.dumps(windows))

    previous_map = None
    for start, end in windows:
        segment_dir = output_dir / f"{start}_{end}"
        trajectory = segment_dir / "mapping_camera_trajectory.csv"
        atlas = segment_dir / "map_atlas.osa"
        if args.resume and trajectory.is_file() and atlas.is_file():
            print(f"[INFO] Reusing completed window {start}_{end}", flush=True)
            previous_map = atlas
            continue
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_video = temporary_dir / f"{start}_{end}.mp4"
        segment_imu = temporary_dir / f"{start}_{end}.json"
        if not os.path.exists(segment_video) or not os.path.exists(segment_imu):
            make_segment_video(args.video, start, end, segment_video)
            load_and_slice_imu(args.imu_json, start / fps, end / fps, segment_imu)
        command = [str(args.slam.resolve()), "-v", str(args.vocabulary.resolve()),
                   "-s", str(args.settings.resolve()), "-i", str(segment_video.resolve()),
                   "-j", str(segment_imu.resolve()), "-o", str(trajectory.resolve()),
                   "--save_map", str(atlas.resolve())]
        if previous_map is not None:
            command += ["--load_map", str(previous_map.resolve())]
        if args.mask_img:
            command += ["--mask_img", str(args.mask_img.resolve())]
        if args.enable_gui:
            command.append("--enable_gui")
        try:
            run(command, cwd=args.slam.resolve().parent,
                stdout_path=segment_dir / "slam_stdout.txt", stderr_path=segment_dir / "slam_stderr.txt")
        except subprocess.CalledProcessError as error:
            sys.exit(f"SLAM failed for window {start}_{end}; see {segment_dir} logs (exit {error.returncode})")
        previous_map = atlas

    merge_trajectories(output_dir, windows, fps, args.stable_frames, args.max_stable_speed)
    print(f"[DONE] Output written to {output_dir}")


if __name__ == "__main__":
    main()
