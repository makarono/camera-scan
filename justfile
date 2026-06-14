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
