# Camera SD card scanner

default: scan

setup:
    #!/usr/bin/env bash
    if [ ! -d .venv ]; then
        uv venv .venv
        uv pip install -r requirements.txt
    fi

# Full scan: YOLOv8s + YOLO-World (slower, catches tractors/pickups)
scan: setup
    .venv/bin/python filter_camera.py

# Fast scan: YOLOv8s only
fast: setup
    .venv/bin/python filter_camera.py --fast

# Delete all results
clean:
    rm -rf results/*/

# Build Docker image
docker-build:
    docker compose build

# Full scan in Docker
docker-scan:
    docker compose run --rm scanner

# Fast scan in Docker
docker-fast:
    docker compose run --rm scanner --fast
