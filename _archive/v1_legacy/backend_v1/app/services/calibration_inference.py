"""
Calibration inference: predict homography from a single pitch image.

Stub implementation returns a constant (identity-like) 3x3 homography until
a trained model is integrated. All config via env; no hardcoded paths.
"""
import base64
import logging
from urllib.request import urlopen

import numpy as np

logger = logging.getLogger(__name__)

# Identity-like homography (row-major) for stub.
STUB_HOMOGRAPHY: list[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def predict_homography(
    image_url: str | None = None,
    image_base64: str | None = None,
) -> tuple[list[float], float | None]:
    """Predict 3x3 homography (flattened to 9-D) from an image.

    Args:
        image_url: URL of the pitch image to fetch.
        image_base64: Base64-encoded image bytes.

    Returns:
        (homography_9, reprojection_error_px). Stub returns constant H and None error.
    """
    if image_url:
        try:
            with urlopen(image_url, timeout=10) as resp:
                _ = resp.read()
        except Exception as e:
            logger.warning("Could not fetch image_url: %s", e)
    elif image_base64:
        try:
            raw = base64.b64decode(image_base64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            _ = arr
        except Exception as e:
            logger.warning("Could not decode image_base64: %s", e)

    # Stub: return constant homography; no reprojection error.
    return (STUB_HOMOGRAPHY.copy(), None)
