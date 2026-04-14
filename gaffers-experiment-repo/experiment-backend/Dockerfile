FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

FROM base AS api
EXPOSE 8100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]

FROM base AS worker
CMD ["python", "worker_main.py"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS worker-gpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gpu.txt requirements-cpu.txt requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements-gpu.txt
COPY . /app
CMD ["python3", "worker_main.py"]
