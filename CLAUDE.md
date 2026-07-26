# camera-scan

Scans surveillance/trail camera SD cards and keeps only files containing people
or vehicles, filtering false triggers (rain, wind, branches). Single Python
script, MegaDetector v6 via ultralytics.

## Development Principles

- **Planning** — use `superpowers:writing-plans` before multi-step work.
  Use `superpowers:brainstorming` when requirements or design are unclear.
- **TDD** — use `superpowers:test-driven-development` before writing
  implementation code. This project has no test suite; see Verification below
  for how changes are proven instead.
- **Debugging** — use `superpowers:systematic-debugging` for any bug or
  unexpected detection result. Do not guess at model behaviour, measure it.
- **Verification** — use `superpowers:verification-before-completion` before
  marking work done.

## Verification

There is no test suite. Prove changes by running the real code path:

```bash
# End-to-end on a synthetic card (no SD card needed)
.venv/bin/python -c "
import cv2, numpy as np, tempfile
from pathlib import Path
import filter_camera as fc
tmp = Path(tempfile.mkdtemp())
card = tmp / 'card'; card.mkdir()
cv2.imwrite(str(card/'IMAG0001.JPG'), (np.random.rand(1080,1920,3)*255).astype('uint8'))
fc.scan_card(fc.load_model(), card, tmp / 'out', no_vlc=True)
print((tmp / 'out' / 'detected.txt').read_text())
"
```

Performance claims must come from measurement on the target machine, never from
parameter counts or published benchmarks. Time `model(frame, ...)` directly.

## Project Facts

- `IMGSZ` must match the resolution the chosen variant was trained at. Raising
  it does not improve detection, it only costs time. `MDV6-yolov10-c` and
  `MDV6-yolov10-e` are 640; `MDV6-yolov10-e-1280` is 1280.
- Model weights come from Zenodo record 15398270. `--model <variant>` downloads
  on demand; a bad name fails with a clean 404 and writes nothing.
- `HALF` (fp16) is enabled on `mps`/`cuda` only. It slows down CPU inference.
- MegaDetector classes are `{0: animal, 1: person, 2: vehicle}`. Only person and
  vehicle are kept.
- Video decode is a significant share of scan wall-clock, not just inference.
  Optimising the model alone has a ceiling.
- Weights (`*.pt`), `.venv/` and `results/` are gitignored.

## Conventions

- Croatian in README and user-facing output; English in code and commit messages.
- Keep the script single-file and dependency-light (`ultralytics`, `opencv-python`).
- Tuning constants live in one block at the top of `filter_camera.py` with a
  comment explaining the value. Keep it that way.
