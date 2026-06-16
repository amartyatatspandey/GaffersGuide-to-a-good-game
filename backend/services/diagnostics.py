import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure a dedicated diagnostics logger
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

diag_handler = logging.FileHandler(LOG_DIR / "diagnostics.log")
diag_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

LOGGER = logging.getLogger("gaffer.diagnostics")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(diag_handler)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

def log_event(category: str, message: str, data: Any = None):
    """Logs a systemic event with metadata."""
    timestamp = datetime.now(timezone.utc).isoformat()
    LOGGER.info(f"[{category}] {message} | data={data}")

def log_error(category: str, error: Exception, context: str = ""):
    """Logs a critical failure with traceback context."""
    LOGGER.error(f"[{category}] FAILURE | context={context} | error={error}", exc_info=True)

def audit_system_imports():
    """Diagnostic check for critical system dependencies."""
    modules = [
        "ultralytics",
        "supervision",
        "cv2",
        "numpy",
        "openai",
        "pydantic",
        "fastapi",
    ]
    missing = []
    for m in modules:
        try:
            __import__(m)
            log_event("AUDIT", f"Module {m} verified.")
        except ImportError:
            missing.append(m)
            log_error("AUDIT", ImportError(f"Module {m} missing."))
    
    return missing
