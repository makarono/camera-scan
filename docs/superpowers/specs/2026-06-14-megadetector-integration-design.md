# MegaDetector v6 Integration — Design

Date: 2026-06-14
Branch: `feature/megadetector-integration`

## Problem

The scanner (`filter_camera.py`) returns many empty videos triggered by wind moving
branches, with no people, animals, or vehicles present. Root causes:

- YOLO-World second pass at low confidence (0.25) hallucinates classes like
  `deer`/`fox`/`person` on foliage, shadows, and shifting light.
- A single sampled frame with any detection flags the whole video.
- Two-pass logic gives clean files a second, more false-positive-prone chance.

This is a classic camera-trap empty-frame problem.

## Goal

Replace the YOLOv8s + YOLO-World two-pass system with a single **MegaDetector v6**
model (`MDV6-yolov10-c`), a model purpose-built to filter empty camera-trap frames.
Keep only **person** and **vehicle** detections (ignore `animal` — wildlife is not wanted).

## Model

- Weights: `MDV6-yolov10-c.pt`
  from `https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1`
- Loadable directly via ultralytics `YOLO()` (confirmed: pytorch-wildlife loads it through
  `ultralytics.models`).
- Class map: `{0: animal, 1: person, 2: vehicle}`. Wanted: `{1: person, 2: vehicle}`.
- Inference image size: **1280** (`IMGSZ` constant).
- Device: MPS on the Apple M2 (already auto-detected).

## Changes to `filter_camera.py`

Remove:
- `WANTED_CLASSES` (COCO mapping), `WORLD_CLASSES`, `WORLD_CONFIDENCE`.
- The two-pass `passes` loop in `scan_card`.
- `--fast` / `--world-only` CLI flags and the `fast` / `world_only` parameters.
- Loading of `yolov8s.pt` and `yolov8s-worldv2.pt`.

Add:
- `MODEL_URL`, `MODEL_WEIGHTS` (local cache path), `IMGSZ = 1280`.
- `WANTED_CLASSES = {1: "person", 2: "vehicle"}`.
- A small download-with-cache helper: if `MODEL_WEIGHTS` is absent, download from
  `MODEL_URL`, then `YOLO(MODEL_WEIGHTS)`.
- Pass `imgsz=IMGSZ` to inference calls.

Keep (already implemented in this branch):
- `find_sd_card`, file pairing, report (`detected.txt`) + playlist (`detected.m3u`), VLC open.
- `process_video`: `grab()`-based frame skipping, batched GPU inference (`BATCH_SIZE`),
  temporal consistency (`MIN_VIDEO_FRAMES = 2`).
- `extract_detections`: minimum bounding-box size filter (`MIN_BOX_AREA_RATIO`).
- `CONFIDENCE_THRESHOLD = 0.25` (MegaDetector recommended range 0.2–0.3).

`scan_card` collapses to a single detection pass over images then videos.

## Supporting files

- `justfile`: collapse `world` / `scan` / `fast` into one `scan` recipe and one
  `docker-scan`; drop the flag-specific recipes.
- `Dockerfile`: pre-download MDv6 weights instead of yolov8 / yolo-world.
- `requirements.txt`: remove `git+https://github.com/ultralytics/CLIP.git` and `ftfy`
  (only needed for YOLO-World text encoding).

## Data flow

SD card → list images + videos → for each file:
- image → one inference (`imgsz=1280`)
- video → skip to every 30th frame via `grab()` → batch on GPU → count person/vehicle
  across sampled frames → require ≥2 frames + min box size

→ if person/vehicle found, log + mark paired file → write report + m3u → open VLC.

## Verification (no automated tests, per user preference)

1. Spike (first step): download weights, assert `YOLO(MODEL_WEIGHTS).names == {0:animal,
   1:person, 2:vehicle}` and a single inference runs on MPS.
2. Run on a real sample folder; compare `detected.txt` count against the old pipeline —
   expect far fewer empty/false-positive videos.

## Risks

- If the `.pt` fails to load directly in ultralytics, fall back to the `pytorch-wildlife`
  package (the spike resolves this before further work).
- Inference cost is tunable via the `IMGSZ` constant if needed.

## Decisions

- Variant: `MDV6-yolov10-c` (compact). The larger `MDV6-yolov10-e-1280` was tried first but
  was too slow on the M2 for large AVI files; the compact model gives a large speedup
  (~6s/video) at a small accuracy cost.
- Video frame sampling uses sequential `cv2.grab()` skipping rather than random
  `CAP_PROP_POS_FRAMES` seeking, which is expensive on compressed AVI.
- Drop `--fast` / `--world-only` rather than keeping them as no-ops (single model makes
  them meaningless).
