"""
Gaffer's Guide FastAPI entry point.

Async-only. Mount routers under /api. Use relative or env-based paths;
no hardcoded local paths.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.routers import calibration, datasets

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks (e.g. load models later)."""
    logger.info("Starting Gaffer's Guide API")
    yield
    logger.info("Shutting down Gaffer's Guide API")


app = FastAPI(
    title="Gaffer's Guide",
    description="Tactical intelligence platform for football CV and analytics",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["calibration"])


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness for Cloud Run."""
    return {"status": "ok"}
