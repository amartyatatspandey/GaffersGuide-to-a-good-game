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