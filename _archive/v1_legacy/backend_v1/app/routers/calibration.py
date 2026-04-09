"""Calibration API: predict homography from a single pitch image."""

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.calibration import CalibrationPredictRequest, CalibrationPredictResponse
from backend.app.services.calibration_inference import predict_homography

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=CalibrationPredictResponse)
async def calibration_predict(request: CalibrationPredictRequest) -> CalibrationPredictResponse:
    """Predict 3x3 homography (flattened to 9-D) from a pitch image URL or base64."""
    try:
        homography, reprojection_error_px = predict_homography(
            image_url=request.image_url,
            image_base64=request.image_base64,
        )
    except Exception as e:
        logger.exception("Calibration predict failed: %s", e)
        raise HTTPException(status_code=500, detail="Calibration prediction failed") from e
    return CalibrationPredictResponse(
        homography=homography,
        reprojection_error_px=reprojection_error_px,
    )
