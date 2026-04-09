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

FROM python:3.11-slim AS worker-gpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-gpu.txt requirements-cpu.txt requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-gpu.txt
COPY . /app
CMD ["python", "worker_main.py"]
