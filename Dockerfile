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

# pre-download MegaDetector v6 weights
RUN .venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1', 'MDV6-yolov10-e-1280.pt')"

COPY filter_camera.py .

ENTRYPOINT [".venv/bin/python", "filter_camera.py", "--no-vlc"]
