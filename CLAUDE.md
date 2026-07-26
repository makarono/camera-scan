# camera-scan

Scans surveillance/trail camera SD cards and keeps only files containing people
or vehicles, filtering false triggers (rain, wind, branches). Single Python
script, MegaDetector v6 via ultralytics.

## Development Principles

- **Planning** — use `superpowers:writing-plans` before multi-step work.
  Use `superpowers:brainstorming` when requirements or design are unclear.
- **TDD** — use `superpowers:test-driven-development` before writing
  implementation code. Detection logic is covered by `test_process_video.py`
  (stub model, no weights or GPU needed). A new check must be shown to fail
  against the previous implementation, or it proves nothing.
- **Debugging** — use `superpowers:systematic-debugging` for any bug or
  unexpected detection result. Do not guess at model behaviour, measure it.
- **Verification** — use `superpowers:verification-before-completion` before
  marking work done.

## Verification

```bash
# Detection logic, fast, no weights needed
.venv/bin/python test_process_video.py
```

Then prove the whole path still runs:

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
- **`MDV6-yolov10-e` is the default, not `-c`.** Measured over the same 242
  files at imgsz 640: `-c` flagged 105, `-e` 89, and the difference is `-c`'s
  false positives. Of 8 sampled files that only `-c` flagged, 7 hold no person
  (the chained fence post with orange tape, up to 72%); of 3 that only `-e`
  flagged, 2 are real and `-c` missed them entirely (a person in night IR, a
  tractor behind branches). Cost: 56 vs 7 ms/frame inference, but decode
  dominates, so the full scan went 7.0 -> 9.9 min.
- **The branch beats `main` on detection, audited in full.** Same 242 files,
  `main` (7fbc11e) flagged 90, the branch 89, 80 shared. Every one of the 19
  disagreements was inspected: all 10 that only `main` flags are the taped
  fence post with nobody there (up to 73%), while 7 of the 9 the branch adds
  are real and `main` missed them outright - a person at 97% and 96%, a tractor
  at 96%. The branch adds 2 new false positives, both under 45%.
- Model weights come from Zenodo record 15398270. `--model <variant>` downloads
  on demand; a bad name fails with a clean 404 and writes nothing.
- `HALF` (fp16) is enabled on `mps`/`cuda` only. It slows down CPU inference.
- MegaDetector classes are `{0: animal, 1: person, 2: vehicle}`. Only person and
  vehicle are kept.
- Video decode is ~2/3 of a clip's cost, inference ~1/3. Optimising the model
  alone has a ceiling; `process_video` streams frames so a confirmed hit stops
  decoding the rest. Files with no detection must still be scanned in full.
- **Camera metadata lies.** The AVIs declare 30 fps but contain ~5.5 fps
  (328 real frames over 60s), and the `movi` chunk declares size 0. Sample by
  `CAP_PROP_POS_MSEC`, never by frame index, or coverage silently collapses to
  one sample every 5.5s. Measured: 2 of 20 clips with people were missed.
- **Seeking is not a shortcut.** These AVIs carry no usable index, so every
  `cap.set(CAP_PROP_POS_MSEC/POS_FRAMES, ...)` makes ffmpeg re-walk the file.
  Measured on three real clips, warm cache, alternating order: 3.3x slower than
  the current grab-walk (5.0-7.8 s vs 1.5-2.3 s per clip), and it lands wrong -
  asking for 1000 ms returns 4367 ms, so the clip's first 4.4 s is never
  sampled. Do not retry this.
- AVFoundation is not a fix for the above. It pads to the declared 30 fps, so
  83% of the frames it returns are duplicates of the previous one. The ffmpeg
  backend returns the real frames.
- Real clips are 3840x2160, ~60s, MJPG, ~49 MB each. Benchmark against these,
  not against synthetic 1080p clips, or the numbers will be optimistic.
- macOS moves files deleted from an SD card to `.Trashes/501` on the card
  itself. Check there before reaching for recovery tools.
- The cameras restart numbering at `IMAG0001` after a card is cleared, so the
  same filename means different recordings in different sessions. Never diff two
  `detected.txt` reports by filename unless both ran over the same input; to
  compare code versions, run both over one identical folder. `detected.txt` also
  lists only positives, so a missing name never means "scanned and rejected".
- Weights (`*.pt`), `.venv/` and `results/` are gitignored.

## Conventions

- Croatian in README and user-facing output; English in code and commit messages.
- Keep the script single-file and dependency-light (`ultralytics`, `opencv-python`).
- Tuning constants live in one block at the top of `filter_camera.py` with a
  comment explaining the value. Keep it that way.
