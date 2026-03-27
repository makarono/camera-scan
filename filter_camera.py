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
RESULTS_DIR = Path("/Users/david/dev/go/src/bitbucket.org/aduroidea/camera/results")


def find_sd_card():
    candidates = []

    # check Docker mount first
    docker_sdcard = Path("/sdcard")
    if docker_sdcard.exists():
        dcim = docker_sdcard / "DCIM"
        if dcim.exists():
            for sub in dcim.iterdir():
                if sub.is_dir() and (any(sub.glob("*.JPG")) or any(sub.glob("*.AVI"))):
                    candidates.append(sub)
            if not candidates:
                candidates.append(dcim)
            return candidates

    # macOS/Linux: scan /Volumes and /media
    for mount_root in [Path("/Volumes"), Path("/media"), Path("/mnt")]:
        if not mount_root.exists():
            continue
        for vol in mount_root.iterdir():
            if not vol.is_dir() or vol.name.startswith("."):
                continue
            dcim = vol / "DCIM"
            if dcim.exists():
                for sub in dcim.iterdir():
                    if sub.is_dir() and (any(sub.glob("*.JPG")) or any(sub.glob("*.AVI"))):
                        candidates.append(sub)
                if not candidates:
                    candidates.append(dcim)
    return candidates


def check_image_yolo(model, path):
    results = model(str(path), verbose=False)[0]
    found = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in WANTED_CLASSES and conf >= CONFIDENCE_THRESHOLD:
            found.append(f"{WANTED_CLASSES[cls_id]}({conf:.0%})")
    return found


def check_image_world(model, path):
    results = model(str(path), verbose=False, conf=WORLD_CONFIDENCE)[0]
    found = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = WORLD_CLASSES[cls_id] if cls_id < len(WORLD_CLASSES) else "unknown"
        found.append(f"{label}({conf:.0%})")
    return found


def check_video_yolo(model, path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {path.name}")
        return []
    found = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % VIDEO_SAMPLE_INTERVAL == 0:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id in WANTED_CLASSES and conf >= CONFIDENCE_THRESHOLD:
                    label = f"{WANTED_CLASSES[cls_id]}({conf:.0%})"
                    if label not in found:
                        found.append(label)
            if found:
                break
        frame_idx += 1
    cap.release()
    return found


def check_video_world(model, path):
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
            results = model(frame, verbose=False, conf=WORLD_CONFIDENCE)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = WORLD_CLASSES[cls_id] if cls_id < len(WORLD_CLASSES) else "unknown"
                tag = f"{label}({conf:.0%})"
                if tag not in found:
                    found.append(tag)
            if found:
                break
        frame_idx += 1
    cap.release()
    return found


def scan_card(yolo_model, world_model, src, out_dir, fast=False, no_vlc=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*"))
    images = [f for f in files if f.suffix.upper() in (".JPG", ".JPEG", ".PNG") and not f.name.startswith(".")]
    videos = [f for f in files if f.suffix.upper() in (".AVI", ".MP4", ".MOV", ".MKV") and not f.name.startswith(".")]

    print(f"Found {len(images)} images, {len(videos)} videos\n")

    results_log = []
    total = len(images) + len(videos)

    # === PASS 1: YOLOv8s ===
    print("--- Pass 1: YOLOv8s ---\n")
    image_hits = set()
    skipped_images = []
    skipped_image_names = set()

    for i, img in enumerate(images, 1):
        print(f"[{i}/{total}] {img.name} ... ", end="", flush=True)
        detected = check_image_yolo(yolo_model, img)
        if detected:
            det_str = ", ".join(detected)
            print(f"YES -> {det_str}")
            results_log.append(f"{img.name}  ->  {det_str}")
            num = int(img.stem.replace("IMAG", ""))
            vid_name = f"IMAG{num+1:04d}.AVI"
            if (img.parent / vid_name).exists():
                results_log.append(f"{vid_name}  ->  {det_str} (from paired image)")
                image_hits.add(vid_name)
        else:
            print("skip")
            skipped_images.append(img)
            skipped_image_names.add(img.name)

    remaining_vids = [v for v in videos if v.name not in image_hits]
    skipped_videos = []
    offset = len(images)
    for i, vid in enumerate(remaining_vids, 1):
        print(f"[{offset+i}/{total}] {vid.name} ... ", end="", flush=True)
        detected = check_video_yolo(yolo_model, vid)
        if detected:
            det_str = ", ".join(detected)
            print(f"YES -> {det_str}")
            results_log.append(f"{vid.name}  ->  {det_str}")
        else:
            print("skip")
            skipped_videos.append(vid)

    # === PASS 2: YOLO-World on skipped files ===
    skipped_total = len(skipped_images) + len(skipped_videos)
    if not fast and skipped_total > 0:
        print(f"\n--- Pass 2: YOLO-World ({skipped_total} skipped files) ---\n")

        for i, img in enumerate(skipped_images, 1):
            print(f"[W {i}/{skipped_total}] {img.name} ... ", end="", flush=True)
            detected = check_image_world(world_model, img)
            if detected:
                det_str = ", ".join(detected)
                print(f"YES -> {det_str} [world]")
                results_log.append(f"{img.name}  ->  {det_str} [world]")
                num = int(img.stem.replace("IMAG", ""))
                vid_name = f"IMAG{num+1:04d}.AVI"
                if (img.parent / vid_name).exists():
                    results_log.append(f"{vid_name}  ->  {det_str} [world] (from paired image)")
                    # remove from skipped_videos if present
                    skipped_videos = [v for v in skipped_videos if v.name != vid_name]
            else:
                print("skip")

        sv_offset = len(skipped_images)
        for i, vid in enumerate(skipped_videos, 1):
            print(f"[W {sv_offset+i}/{skipped_total}] {vid.name} ... ", end="", flush=True)
            detected = check_video_world(world_model, vid)
            if detected:
                det_str = ", ".join(detected)
                print(f"YES -> {det_str} [world]")
                results_log.append(f"{vid.name}  ->  {det_str} [world]")
            else:
                print("skip")

    # build paired playlist entries
    positive_files = []
    seen = set()
    for line in sorted(results_log):
        name = line.split("  ->")[0]
        if name in seen:
            continue
        seen.add(name)
        path = src / name
        if path.exists():
            positive_files.append((name, path))

    playlist_entries = []
    added = set()
    for name, path in positive_files:
        if name in added:
            continue
        num = int(Path(name).stem.replace("IMAG", ""))
        if name.upper().endswith(".JPG"):
            jpg = path
            avi = src / f"IMAG{num+1:04d}.AVI"
            playlist_entries.append(jpg)
            added.add(jpg.name)
            if avi.exists():
                playlist_entries.append(avi)
                added.add(avi.name)
        elif name.upper().endswith(".AVI"):
            jpg = src / f"IMAG{num-1:04d}.JPG"
            if jpg.exists() and jpg.name not in added:
                playlist_entries.append(jpg)
                added.add(jpg.name)
            playlist_entries.append(path)
            added.add(name)

    # write report
    report = out_dir / "detected.txt"
    with open(report, "w") as f:
        f.write(f"Detected files: {len(results_log)}\n")
        f.write(f"Source: {src}\n")
        f.write(f"Scanned: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Models: {'YOLOv8s only' if fast else 'YOLOv8s + YOLO-World'}\n\n")
        for line in sorted(results_log):
            f.write(line + "\n")

    # write playlist
    playlist = out_dir / "detected.m3u"
    with open(playlist, "w") as f:
        f.write("#EXTM3U\n")
        for entry in playlist_entries:
            f.write(f"#EXTINF:-1,{entry.name}\n")
            f.write(f"{entry}\n")

    print(f"\nDone! {len(results_log)} detections -> {report}")
    print(f"Playlist: {playlist} ({len(playlist_entries)} files)")

    # open VLC
    if playlist_entries and not no_vlc:
        print("Opening VLC in 5 seconds...")
        time.sleep(5)
        vlc = Path("/Applications/VLC.app/Contents/MacOS/VLC")
        if vlc.exists():
            subprocess.Popen([str(vlc), str(playlist)])
        else:
            subprocess.Popen(["open", "-a", "VLC", str(playlist)])
        print("VLC opened")

    return len(playlist_entries) > 0


def main():
    parser = argparse.ArgumentParser(description="Camera SD card scanner")
    parser.add_argument("--fast", action="store_true", help="Fast scan: YOLOv8s only, skip YOLO-World")
    parser.add_argument("--no-vlc", action="store_true", help="Skip opening VLC (for Docker)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = "FAST (YOLOv8s only)" if args.fast else "FULL (YOLOv8s + YOLO-World)"
    print(f"Mode: {mode}")
    print("Loading models...")
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
        if candidates:
            print("\nSD cards found:")
            for i, c in enumerate(candidates, 1):
                count = len(list(c.glob("*")))
                print(f"  {i}. {c} ({count} files)")
            if len(candidates) == 1:
                src = candidates[0]
                print(f"\nUsing: {src}")
            else:
                choice = input("\nSelect card number: ").strip()
                src = candidates[int(choice) - 1]
        else:
            print("\nNo SD card with DCIM found!")
            input("Insert SD card and press Enter...")
            continue

        camera_name = input("Camera name/label (eg. front, barn, gate): ").strip()
        if not camera_name:
            camera_name = "unknown"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = RESULTS_DIR / camera_name / timestamp

        print(f"\nScanning: {src}")
        print(f"Results:  {out_dir}\n")

        scan_card(yolo_model, world_model, src, out_dir, fast=args.fast, no_vlc=args.no_vlc)

        print("\n" + "-" * 40)
        again = input("Scan another card? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nAll results saved in:", RESULTS_DIR)
    print("Bye!")


if __name__ == "__main__":
    main()
