FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
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
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Copy local code to the container image.
COPY . ./

# Install production dependencies.
# AI Note: Ensure requirements.txt uses pinned versions (e.g., pandas==2.0.3)
RUN pip install --no-cache-dir -r requirements.txt

# Backend app is backend.app.main:app; PYTHONPATH so "backend" is importable.
ENV PYTHONPATH=/app

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker backend.app.main:app