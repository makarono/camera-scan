#!/usr/bin/env python3
"""Filter surveillance camera files - keep only those with people, animals, or vehicles.
Two-pass detection: YOLOv8s first, then YOLO-World on skipped files for tractor/pickup."""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import cv2

# Pass 1: YOLOv8s COCO classes
WANTED_CLASSES = {
    0: "person",
    1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

# Pass 2: YOLO-World custom classes
WORLD_CLASSES = [
    "person", "car", "truck", "pickup truck", "tractor",
    "motorcycle", "bicycle", "bus",
    "dog", "cat", "horse", "cow", "sheep", "bear",
    "deer", "wild boar", "fox",
]

CONFIDENCE_THRESHOLD = 0.35
WORLD_CONFIDENCE = 0.25  # lower threshold for second pass
VIDEO_SAMPLE_INTERVAL = 30
# Default to 'results' folder in the same directory as the script
RESULTS_DIR = Path(__file__).parent / "results"


def find_sd_card():
    """Find SD cards with DCIM folders on macOS, Linux or Docker."""
    roots = [Path("/sdcard"), Path("/Volumes"), Path("/media"), Path("/mnt")]
    candidates = []

    for root in roots:
        if not root.exists():
            continue

        # If the root itself is the SD card (like in Docker)
        search_dirs = [root] if root.name == "sdcard" else root.iterdir()

        for vol in search_dirs:
            if not vol.is_dir() or vol.name.startswith("."):
                continue
            dcim = vol / "DCIM"
            if dcim.exists():
                # Check for subdirectories with media
                subdirs = [d for d in dcim.iterdir() if d.is_dir() and not d.name.startswith(".")]
                for sub in subdirs:
                    if any(sub.glob("*.JPG")) or any(sub.glob("*.AVI")):
                        candidates.append(sub)
                if not subdirs:
                    candidates.append(dcim)

    return candidates


def check_media(model, path_or_frame, classes, conf):
    """Run detection on an image path or a video frame."""
    results = model(path_or_frame, verbose=False, conf=conf)[0]
    found = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])

        if isinstance(classes, dict):
            if cls_id not in classes:
                continue
            label = classes[cls_id]
        else:
            label = classes[cls_id] if cls_id < len(classes) else "unknown"

        found.append(f"{label}({conf_val:.0%})")
    return list(dict.fromkeys(found))  # remove duplicates


def process_video(model, path, classes, conf):
    """Sample frames from video and run detection."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    found = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % VIDEO_SAMPLE_INTERVAL == 0:
            found = check_media(model, frame, classes, conf)
            if found:
                break
        frame_idx += 1
    cap.release()
    return found


def get_paired_file(file_path):
    """Get the paired file (JPG for AVI or vice versa)."""
    stem = file_path.stem
    if not stem.startswith("IMAG"):
        return None

    try:
        num = int(stem.replace("IMAG", ""))
    except ValueError:
        return None

    if file_path.suffix.upper() == ".JPG":
        paired = file_path.parent / f"IMAG{num+1:04d}.AVI"
    else:
        paired = file_path.parent / f"IMAG{num-1:04d}.JPG"

    return paired if paired.exists() else None


def scan_card(yolo_model, world_model, src, out_dir, fast=False, no_vlc=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.glob("*") if not f.name.startswith("."))
    images = [f for f in files if f.suffix.upper() in (".JPG", ".JPEG", ".PNG")]
    videos = [f for f in files if f.suffix.upper() in (".AVI", ".MP4", ".MOV", ".MKV")]

    print(f"Found {len(images)} images, {len(videos)} videos\n")

    results_log = {} # filename -> detection_string
    total = len(images) + len(videos)

    passes = [
        ("YOLOv8s", yolo_model, WANTED_CLASSES, CONFIDENCE_THRESHOLD),
    ]
    if not fast:
        passes.append(("YOLO-World", world_model, WORLD_CLASSES, WORLD_CONFIDENCE))

    for pass_name, model, classes, conf in passes:
        print(f"--- Pass: {pass_name} ---")

        # Filter files that haven't been detected yet
        to_process = [f for f in images + videos if f.name not in results_log]
        if not to_process:
            continue

        for i, f in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] {f.name} ... ", end="", flush=True)

            # Check paired image first if this is a video
            if f.suffix.upper() in (".AVI", ".MP4", ".MOV", ".MKV"):
                paired_img = get_paired_file(f)
                if paired_img and paired_img.name in results_log:
                    det_str = results_log[paired_img.name]
                    print(f"YES -> {det_str} (from paired image)")
                    results_log[f.name] = det_str
                    continue

            detected = process_video(model, f, classes, conf) if f.suffix.upper() in (".AVI", ".MP4", ".MOV", ".MKV") \
                       else check_media(model, f, classes, conf)

            if detected:
                det_str = ", ".join(detected)
                print(f"YES -> {det_str}")
                results_log[f.name] = det_str

                # If it's an image, also mark the paired video
                if f.suffix.upper() == ".JPG":
                    paired_vid = get_paired_file(f)
                    if paired_vid:
                        results_log[paired_vid.name] = det_str
            else:
                print("skip")
        print()

    # Prepare playlist
    playlist_files = []
    added = set()
    for name in sorted(results_log.keys()):
        f = src / name
        if name not in added and f.exists():
            playlist_files.append(f)
            added.add(name)
            # Ensure both of the pair are in the playlist together
            paired = get_paired_file(f)
            if paired and paired.name not in added:
                playlist_files.append(paired)
                added.add(paired.name)

    # Sort playlist to keep pairs together
    playlist_files.sort(key=lambda x: x.name)

    # Write report
    report = out_dir / "detected.txt"
    with open(report, "w") as f:
        f.write(f"Detected files: {len(results_log)}\n")
        f.write(f"Source: {src}\n")
        f.write(f"Scanned: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Models: {'YOLOv8s only' if fast else 'YOLOv8s + YOLO-World'}\n\n")
        for name, det in sorted(results_log.items()):
            f.write(f"{name}  ->  {det}\n")

    # Write playlist
    playlist = out_dir / "detected.m3u"
    with open(playlist, "w") as f:
        f.write("#EXTM3U\n")
        for entry in playlist_files:
            f.write(f"#EXTINF:-1,{entry.name}\n")
            f.write(f"{entry}\n")

    print(f"Done! {len(results_log)} detections -> {report}")
    print(f"Playlist: {playlist} ({len(playlist_files)} files)")

    if playlist_files and not no_vlc:
        print("Opening VLC in 5 seconds...")
        time.sleep(5)
        vlc = Path("/Applications/VLC.app/Contents/MacOS/VLC")
        subprocess.Popen([str(vlc) if vlc.exists() else "vlc", str(playlist)])

    return len(results_log) > 0


def main():
    parser = argparse.ArgumentParser(description="Camera SD card scanner")
    parser.add_argument("--fast", action="store_true", help="Fast scan: YOLOv8s only, skip YOLO-World")
    parser.add_argument("--no-vlc", action="store_true", help="Skip opening VLC (for Docker)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = "FAST (YOLOv8s only)" if args.fast else "FULL (YOLOv8s + YOLO-World)"
    print(f"Mode: {mode}\nLoading models...")

    yolo_model = YOLO("yolov8s.pt")
    world_model = None
    if not args.fast:
        world_model = YOLO("yolov8s-worldv2.pt")
        world_model.set_classes(WORLD_CLASSES)
    print("Models ready\n")

    while True:
        print("\n" + "=" * 60)
        print(f"CAMERA SD CARD SCANNER [{mode}]")
        print("=" * 60)

        candidates = find_sd_card()
        if not candidates:
            print("\nNo SD card with DCIM found!")
            if input("Insert SD card and press Enter (or 'q' to quit)...").lower() == 'q':
                break
            continue

        print("\nSD cards found:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c} ({len(list(c.glob('*')))} files)")

        choice = "1" if len(candidates) == 1 else input("\nSelect card number: ").strip()
        try:
            src = candidates[int(choice) - 1]
        except (ValueError, IndexError):
            continue

        camera_name = input("Camera name/label (eg. front, barn, gate): ").strip() or "unknown"
        out_dir = RESULTS_DIR / camera_name / datetime.now().strftime("%Y-%m-%d_%H-%M")

        print(f"\nScanning: {src}\nResults:  {out_dir}\n")
        scan_card(yolo_model, world_model, src, out_dir, fast=args.fast, no_vlc=args.no_vlc)

        if input("\nScan another card? (y/n): ").strip().lower() != "y":
            break

    print(f"\nAll results saved in: {RESULTS_DIR}\nBye!")


if __name__ == "__main__":
    main()
