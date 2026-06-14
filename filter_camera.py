#!/usr/bin/env python3
"""Filter trail/surveillance camera files - keep only those with people or vehicles.
Single-pass detection with MegaDetector v6 (MDV6-yolov10-c), a camera-trap model."""

import argparse
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1"
MODEL_WEIGHTS = Path(__file__).parent / "MDV6-yolov10-c.pt"

# MegaDetector v6 classes are {0: animal, 1: person, 2: vehicle}; keep only person + vehicle.
WANTED_CLASSES = {1: "person", 2: "vehicle"}

CONFIDENCE_THRESHOLD = 0.25  # MegaDetector recommended range 0.2-0.3
IMGSZ = 1280  # MegaDetector v6 inference resolution
VIDEO_SAMPLE_INTERVAL = 30
BATCH_SIZE = 16  # frames per GPU inference batch
MIN_VIDEO_FRAMES = 2  # object must appear in this many sampled frames (kills wind/branch false positives)
MIN_BOX_AREA_RATIO = 0.004  # ignore tiny detections (foliage/shadow noise), fraction of frame area
# Default to 'results' folder in the same directory as the script
RESULTS_DIR = Path(__file__).parent / "results"


def load_model():
    """Download MegaDetector v6 weights if missing, then load via ultralytics."""
    if not MODEL_WEIGHTS.exists():
        print(f"Downloading MegaDetector weights -> {MODEL_WEIGHTS.name} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_WEIGHTS)
    return YOLO(str(MODEL_WEIGHTS))


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


def extract_detections(results, classes):
    """Turn one Results object into a list of 'label(conf%)' strings, filtering tiny boxes."""
    h, w = results.orig_shape
    frame_area = float(h * w)
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

        x1, y1, x2, y2 = box.xyxy[0]
        box_area = float((x2 - x1) * (y2 - y1))
        if frame_area and box_area / frame_area < MIN_BOX_AREA_RATIO:
            continue

        found.append(f"{label}({conf_val:.0%})")
    return list(dict.fromkeys(found))  # remove duplicates


def check_media(model, path_or_frame, classes, conf):
    """Run detection on an image path or a video frame."""
    results = model(path_or_frame, verbose=False, conf=conf, device=DEVICE, imgsz=IMGSZ)[0]
    return extract_detections(results, classes)


def process_video(model, path, classes, conf):
    """Sample frames from video and run batched detection. Early-exits per batch."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    # Advance with grab() (cheap, no color convert) and only decode every Nth frame.
    frames = []
    idx = 0
    while cap.grab():
        if idx % VIDEO_SAMPLE_INTERVAL == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        return []

    label_frames = {}  # base label -> number of sampled frames it appeared in
    best_str = {}      # base label -> display string (last seen)
    for start in range(0, len(frames), BATCH_SIZE):
        batch = frames[start:start + BATCH_SIZE]
        for results in model(batch, verbose=False, conf=conf, device=DEVICE, imgsz=IMGSZ):
            for d in extract_detections(results, classes):
                base = d.split("(")[0]
                label_frames[base] = label_frames.get(base, 0) + 1
                best_str[base] = d
        confirmed = [b for b, n in label_frames.items() if n >= MIN_VIDEO_FRAMES]
        if confirmed:
            return [best_str[b] for b in confirmed]

    return []


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


def scan_card(model, src, out_dir, no_vlc=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.glob("*") if not f.name.startswith("."))
    images = [f for f in files if f.suffix.upper() in (".JPG", ".JPEG", ".PNG")]
    videos = [f for f in files if f.suffix.upper() in (".AVI", ".MP4", ".MOV", ".MKV")]

    print(f"Found {len(images)} images, {len(videos)} videos\n")

    results_log = {}  # filename -> detection_string
    all_files = images + videos

    for i, f in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] {f.name} ... ", end="", flush=True)
        is_video = f.suffix.upper() in (".AVI", ".MP4", ".MOV", ".MKV")

        # Reuse a paired image's result for its video
        if is_video:
            paired_img = get_paired_file(f)
            if paired_img and paired_img.name in results_log:
                det_str = results_log[paired_img.name]
                print(f"YES -> {det_str} (from paired image)")
                results_log[f.name] = det_str
                continue

        detected = process_video(model, f, WANTED_CLASSES, CONFIDENCE_THRESHOLD) if is_video \
                   else check_media(model, f, WANTED_CLASSES, CONFIDENCE_THRESHOLD)

        if detected:
            det_str = ", ".join(detected)
            print(f"YES -> {det_str}")
            results_log[f.name] = det_str
            if not is_video:
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
        f.write("Models: MegaDetector v6 (MDV6-yolov10-c)\n\n")
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
    parser.add_argument("--no-vlc", action="store_true", help="Skip opening VLC (for Docker)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MegaDetector v6...")
    model = load_model()
    print("Model ready\n")

    while True:
        print("\n" + "=" * 60)
        print("CAMERA SD CARD SCANNER [MegaDetector v6]")
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
        scan_card(model, src, out_dir, no_vlc=args.no_vlc)

        if input("\nScan another card? (y/n): ").strip().lower() != "y":
            break

    print(f"\nAll results saved in: {RESULTS_DIR}\nBye!")


if __name__ == "__main__":
    main()
