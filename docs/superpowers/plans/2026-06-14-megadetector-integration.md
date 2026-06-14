# MegaDetector v6 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the YOLOv8s + YOLO-World two-pass detector in `filter_camera.py` with a single MegaDetector v6 (`MDV6-yolov10-e`) pass that keeps only person + vehicle, killing wind/vegetation false positives.

**Architecture:** One ultralytics-loaded model. Weights auto-downloaded from Zenodo and cached next to the script. Existing scaffolding (SD discovery, pairing, batched/temporal video sampling, min-box filter, report/playlist, VLC) is kept; the two-pass loop, COCO/World class maps, and `--fast`/`--world-only` flags are removed.

**Tech Stack:** Python 3.14, ultralytics 8.4.30, torch 2.11 (MPS), opencv-python.

**Note on tests:** Per user preference there are no automated tests; verification is runtime (`py_compile`, model load check, real-folder run).

---

## File Structure

- Modify: `filter_camera.py` — model loading, constants, single-pass scan, CLI.
- Modify: `justfile` — collapse recipes to `scan` / `docker-scan`.
- Modify: `Dockerfile` — pre-download MDv6 weights.
- Modify: `requirements.txt` — drop CLIP + ftfy.
- Modify: `.gitignore` — ignore weight files.

---

### Task 1: Feasibility spike — confirm MDv6 loads via ultralytics

**Files:** none (throwaway check)

- [ ] **Step 1: Download weights and verify load + class names + inference**

Run:
```bash
.venv/bin/python -c "
import urllib.request
from pathlib import Path
from ultralytics import YOLO
import torch
w = Path('MDV6-yolov10-e-1280.pt')
if not w.exists():
    urllib.request.urlretrieve('https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1', w)
m = YOLO(str(w))
print('names:', m.names)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
import numpy as np
r = m(np.zeros((1280,1280,3), dtype='uint8'), imgsz=1280, device=dev, verbose=False)
print('inference OK on', dev, '-> boxes:', len(r[0].boxes))
"
```
Expected: `names: {0: 'animal', 1: 'person', 2: 'vehicle'}` and `inference OK on mps -> boxes: 0`.

- [ ] **Step 2: Decide fallback**

If load fails, stop and switch to the `pytorch-wildlife` package (`pip install PytorchWildlife`, use `MegaDetectorV6(version='MDV6-yolov10-e')`). Otherwise continue — the direct path works.

---

### Task 2: Rewrite constants, imports, and add `load_model`

**Files:**
- Modify: `filter_camera.py:1-35` (docstring, imports, constants)

- [ ] **Step 1: Replace module docstring + imports**

Replace lines 1-14 with:
```python
#!/usr/bin/env python3
"""Filter trail/surveillance camera files - keep only those with people or vehicles.
Single-pass detection with MegaDetector v6 (MDV6-yolov10-e), a camera-trap model."""

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
```

- [ ] **Step 2: Replace the class maps + constants block**

Replace the old `WANTED_CLASSES` / `WORLD_CLASSES` / thresholds block (lines 16-35) with:
```python
MODEL_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1"
MODEL_WEIGHTS = Path(__file__).parent / "MDV6-yolov10-e-1280.pt"

# MegaDetector v6 classes are {0: animal, 1: person, 2: vehicle}; keep only person + vehicle.
WANTED_CLASSES = {1: "person", 2: "vehicle"}

CONFIDENCE_THRESHOLD = 0.25  # MegaDetector recommended range 0.2-0.3
IMGSZ = 1280  # MDV6-yolov10-e is the 1280px variant
VIDEO_SAMPLE_INTERVAL = 30
BATCH_SIZE = 16  # frames per GPU inference batch
MIN_VIDEO_FRAMES = 2  # object must appear in this many sampled frames (kills wind/branch false positives)
MIN_BOX_AREA_RATIO = 0.004  # ignore tiny detections (foliage/shadow noise), fraction of frame area
RESULTS_DIR = Path(__file__).parent / "results"


def load_model():
    """Download MegaDetector v6 weights if missing, then load via ultralytics."""
    if not MODEL_WEIGHTS.exists():
        print(f"Downloading MegaDetector weights -> {MODEL_WEIGHTS.name} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_WEIGHTS)
    return YOLO(str(MODEL_WEIGHTS))
```

- [ ] **Step 3: Verify it compiles**

Run: `.venv/bin/python -m py_compile filter_camera.py`
Expected: no output (success). Note: `scan_card`/`main` still reference removed names — that's fixed in Tasks 4-5; this step only checks the edited block parses.

---

### Task 3: Pass `imgsz` to inference calls

**Files:**
- Modify: `filter_camera.py` (`check_media`, `process_video`)

- [ ] **Step 1: Update `check_media` inference call**

Change:
```python
    results = model(path_or_frame, verbose=False, conf=conf, device=DEVICE)[0]
```
to:
```python
    results = model(path_or_frame, verbose=False, conf=conf, device=DEVICE, imgsz=IMGSZ)[0]
```

- [ ] **Step 2: Update `process_video` batch inference call**

Change:
```python
        for results in model(batch, verbose=False, conf=conf, device=DEVICE):
```
to:
```python
        for results in model(batch, verbose=False, conf=conf, device=DEVICE, imgsz=IMGSZ):
```

- [ ] **Step 3: Verify compile**

Run: `.venv/bin/python -m py_compile filter_camera.py`
Expected: no output.

---

### Task 4: Collapse `scan_card` to a single pass

**Files:**
- Modify: `filter_camera.py:152-205` (function signature through the pass loop)

- [ ] **Step 1: Replace the signature and the two-pass loop**

Replace from `def scan_card(...)` down to (and including) the `passes` loop and its trailing `print()` — i.e. the current lines 152-205 — with:
```python
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
```

- [ ] **Step 2: Update the report's Models line**

In the report-writing block, change:
```python
        f.write(f"Models: {'YOLO-World only' if world_only else ('YOLOv8s only' if fast else 'YOLOv8s + YOLO-World')}\n\n")
```
to:
```python
        f.write("Models: MegaDetector v6 (MDV6-yolov10-e)\n\n")
```

- [ ] **Step 3: Verify compile**

Run: `.venv/bin/python -m py_compile filter_camera.py`
Expected: no output. (`main` still calls the old signature — fixed next task.)

---

### Task 5: Simplify `main()`

**Files:**
- Modify: `filter_camera.py:254-308` (`main` body)

- [ ] **Step 1: Replace argument parsing + model loading**

Replace:
```python
    parser = argparse.ArgumentParser(description="Camera SD card scanner")
    parser.add_argument("--fast", action="store_true", help="Fast scan: YOLOv8s only, skip YOLO-World")
    parser.add_argument("--world-only", action="store_true", help="World-only scan: YOLO-World only, skip YOLOv8s")
    parser.add_argument("--no-vlc", action="store_true", help="Skip opening VLC (for Docker)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fast and args.world_only:
        print("Error: --fast and --world-only are mutually exclusive")
        return

    mode = "WORLD ONLY (YOLO-World)" if args.world_only else ("FAST (YOLOv8s only)" if args.fast else "FULL (YOLOv8s + YOLO-World)")
    print(f"Mode: {mode}\nLoading models...")

    yolo_model = None if args.world_only else YOLO("yolov8s.pt")
    world_model = None
    if not args.fast:
        world_model = YOLO("yolov8s-worldv2.pt")
        world_model.set_classes(WORLD_CLASSES)
    print("Models ready\n")
```
with:
```python
    parser = argparse.ArgumentParser(description="Camera SD card scanner")
    parser.add_argument("--no-vlc", action="store_true", help="Skip opening VLC (for Docker)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MegaDetector v6...")
    model = load_model()
    print("Model ready\n")
```

- [ ] **Step 2: Update the banner line**

Change:
```python
        print(f"CAMERA SD CARD SCANNER [{mode}]")
```
to:
```python
        print("CAMERA SD CARD SCANNER [MegaDetector v6]")
```

- [ ] **Step 3: Update the scan_card call**

Change:
```python
        scan_card(yolo_model, world_model, src, out_dir, fast=args.fast, world_only=args.world_only, no_vlc=args.no_vlc)
```
to:
```python
        scan_card(model, src, out_dir, no_vlc=args.no_vlc)
```

- [ ] **Step 4: Verify compile + no leftover references**

Run:
```bash
.venv/bin/python -m py_compile filter_camera.py && \
! grep -nE "WORLD_CLASSES|WORLD_CONFIDENCE|world_only|world_model|yolov8s" filter_camera.py && echo CLEAN
```
Expected: `CLEAN`.

- [ ] **Step 5: Commit**

```bash
git add filter_camera.py
git commit -m "Replace YOLOv8s+YOLO-World with single MegaDetector v6 pass"
```

---

### Task 6: Update justfile, Dockerfile, requirements, gitignore

**Files:**
- Modify: `justfile`, `Dockerfile`, `requirements.txt`, `.gitignore`

- [ ] **Step 1: Rewrite `justfile`**

Replace the whole file with:
```make
# Camera SD card scanner

default: scan

setup:
    #!/usr/bin/env bash
    if [ ! -d .venv ]; then
        uv venv .venv
        uv pip install -r requirements.txt
    fi

# Scan SD card with MegaDetector v6
scan: setup
    .venv/bin/python filter_camera.py

# Delete all results
clean:
    rm -rf results/*/

# Build Docker image
docker-build:
    docker compose build

# Scan SD card in Docker
docker-scan:
    docker compose run --rm scanner
```

- [ ] **Step 2: Update `Dockerfile` model pre-download**

Replace:
```dockerfile
# pre-download models
RUN .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8s-worldv2.pt')"
```
with:
```dockerfile
# pre-download MegaDetector v6 weights
RUN .venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1', 'MDV6-yolov10-e-1280.pt')"
```

- [ ] **Step 3: Rewrite `requirements.txt`**

Replace the whole file with:
```
ultralytics
opencv-python
```

- [ ] **Step 4: Ignore weight files in `.gitignore`**

Append this line to `.gitignore` if not already present:
```
*.pt
```

- [ ] **Step 5: Commit**

```bash
git add justfile Dockerfile requirements.txt .gitignore
git commit -m "Update tooling for single MegaDetector v6 model"
```

---

### Task 7: Runtime verification on a real sample

**Files:** none

- [ ] **Step 1: Run against a folder of real camera files**

Run (pick a folder that has known empty wind videos plus at least one with a person/vehicle):
```bash
.venv/bin/python -c "
from pathlib import Path
import filter_camera as fc
fc.scan_card(fc.load_model(), Path('PATH/TO/SAMPLE'), Path('results/_verify'), no_vlc=True)
"
cat results/_verify/detected.txt
```
Expected: empty wind/branch videos are NOT listed; person/vehicle clips ARE listed. Far fewer entries than the old pipeline.

- [ ] **Step 2: Tune if needed**

If still too many false positives: raise `MIN_VIDEO_FRAMES` to 3 or `CONFIDENCE_THRESHOLD` to 0.3. If real events are missed: lower `CONFIDENCE_THRESHOLD` to 0.2 or `MIN_BOX_AREA_RATIO` to 0.002. Re-run Step 1.

- [ ] **Step 3: Clean up verify output**

```bash
rm -rf results/_verify
```

---

## Self-Review

- **Spec coverage:** model swap (T2), person+vehicle filter (T2), imgsz=1280 (T2-T3), single pass (T4), flag removal (T5), justfile/Dockerfile/requirements (T6), verification (T1, T7). All spec sections covered.
- **Placeholder scan:** sample path in T7 Step 1 is intentionally user-supplied; all code steps contain full code.
- **Type consistency:** `load_model()`, `scan_card(model, src, out_dir, no_vlc=...)`, `WANTED_CLASSES` dict, `IMGSZ` used consistently across tasks.
