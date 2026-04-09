FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# libgl1: OpenCV GUI/codec paths; libgl1-mesa-glx is absent on newer Debian/Ubuntu bases.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app

ENV PYTHONPATH=/app/backend:/app

# Default command keeps container reusable for batch jobs.
CMD ["python", "backend/scripts/cloud_batch_processor.py"]