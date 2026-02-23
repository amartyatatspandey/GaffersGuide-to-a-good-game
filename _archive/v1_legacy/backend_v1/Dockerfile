# Backend image for Gaffer's Guide API (Cloud Run).
# Build from repo root: docker build -f backend/Dockerfile .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Copy repo so backend package is at /app/backend.
COPY . ./

# Install backend dependencies.
RUN pip install --no-cache-dir -r backend/requirements.txt

# Backend app is backend.app.main:app; PYTHONPATH so "backend" is importable.
ENV PYTHONPATH=/app

CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker backend.app.main:app
