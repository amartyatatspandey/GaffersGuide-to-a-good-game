"""Pydantic models for calibration inference API."""

from pydantic import BaseModel, Field, model_validator


class CalibrationPredictRequest(BaseModel):
    """Request for calibration prediction: provide image via URL or base64."""

    image_url: str | None = Field(None, description="URL of the pitch image")
    image_base64: str | None = Field(None, description="Base64-encoded image data")

    @model_validator(mode="after")
    def at_least_one_source(self) -> "CalibrationPredictRequest":
        if not self.image_url and not self.image_base64:
            raise ValueError("Provide either image_url or image_base64")
        return self


class CalibrationPredictResponse(BaseModel):
    """Calibration prediction: 3x3 homography flattened to 9 values, optional reprojection error."""

    homography: list[float] = Field(..., min_length=9, max_length=9, description="Row-major 3x3 homography")
    reprojection_error_px: float | None = Field(None, description="Reprojection error in pixels if available")
