# Camera SD card scanner

default: world

setup:
    #!/usr/bin/env bash
    if [ ! -d .venv ]; then
        uv venv .venv
        uv pip install -r requirements.txt
    fi

# World-only scan: YOLO-World only (default)
world: setup
    .venv/bin/python filter_camera.py --world-only

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

# World-only scan in Docker (default)
docker-world:
    docker compose run --rm scanner --world-only

# Full scan in Docker
docker-scan:
    docker compose run --rm scanner

# Fast scan in Docker
docker-fast:
    docker compose run --rm scanner --fast
