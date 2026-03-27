FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .
RUN uv venv .venv && uv pip install -r requirements.txt

# pre-download models
RUN .venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8s-worldv2.pt')"

COPY filter_camera.py .

ENTRYPOINT [".venv/bin/python", "filter_camera.py", "--no-vlc"]
