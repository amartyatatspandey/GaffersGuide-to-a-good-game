FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Full tree copy includes backend/references/sn-calibration if present in build context.
# Homography auto-generation requires that checkout plus resources/ (see Development_Protocol.md §8).
# For smaller images, omit it and mount weights at runtime or pre-generate *_homographies.json.
COPY . /app

# Symlink data, output, and model weight directories to the GCS FUSE mount path
RUN rm -rf /app/backend/data/uploads /app/backend/data/reports /app/backend/output /app/backend/models/pretrained && \
    mkdir -p /app/backend/data /app/backend/models && \
    ln -s /mnt/gaffers-storage/uploads /app/backend/data/uploads && \
    ln -s /mnt/gaffers-storage/reports /app/backend/data/reports && \
    ln -s /mnt/gaffers-storage/output /app/backend/output && \
    ln -s /mnt/gaffers-storage/models /app/backend/models/pretrained

ENV PYTHONPATH=/app/backend:/app
EXPOSE 8000


CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
